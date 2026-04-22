import os
import argparse
import csv
import html
import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLE_PDX_EAGER_INIT", "False")
os.environ.setdefault("FLAGS_use_mkldnn", "0")

from llm.refine import load_prompt, normalize_final_invoice_json, refine_invoice_json_with_llm
from ocr.ocr_pipeline import run_multi_ocr
from preprocessing.preprocess import preprocess_invoice


LOGGER = logging.getLogger("batch_pipeline")

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _log_step(
    index: int,
    total: int,
    invoice_type: str,
    file_path: Path,
    step: str,
    status: str,
    extra: str = "",
) -> None:
    suffix = f" | {extra}" if extra else ""
    msg = (
        f"[{index}/{total}] {step.upper()} {status.upper()} | "
        f"type={invoice_type} | file={str(file_path)}{suffix}"
    )
    LOGGER.info(msg)
    print(msg, flush=True)


def _discover_files(folder: Path) -> List[Path]:
    files: List[Path] = []
    if not folder.exists() or not folder.is_dir():
        return files
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return files


def _safe_json_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_text_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _extract_admin_fields(final_json: Dict[str, Any]) -> Dict[str, Any]:
    products = final_json.get("productsArray", [])
    expenses = final_json.get("Expenses", [])
    return {
        "invoice_number": final_json.get("Invoice_Number"),
        "invoice_date": final_json.get("Invoice_Date"),
        "supplier_name": final_json.get("SupplierName"),
        "customer_name": final_json.get("Customer_Name"),
        "supplier_gst": final_json.get("Supplier_GST"),
        "customer_gst": final_json.get("Customer_GSTIN"),
        "taxable_amount": final_json.get("Taxable_Amount"),
        "total_amount": final_json.get("Total_Amount"),
        "products_count": len(products) if isinstance(products, list) else 0,
        "expenses_count": len(expenses) if isinstance(expenses, list) else 0,
    }


def _prepare_llm_context(ocr_debug: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    paddle_payload = ocr_debug.get("paddle", {}) if isinstance(ocr_debug, dict) else {}
    paddle_json = paddle_payload.get("json") if isinstance(paddle_payload, dict) else {}
    paddle_text = paddle_payload.get("text") if isinstance(paddle_payload, dict) else ""
    tess_payload = ocr_debug.get("tesseract", {}) if isinstance(ocr_debug, dict) else {}
    tesseract_text = tess_payload.get("text") if isinstance(tess_payload, dict) else ""
    raw_ocr_context = {
        "paddle_json": paddle_json,
        "paddle_text": paddle_text,
        "tesseract_text": tesseract_text,
        "tesseract_boxes": tess_payload.get("boxes") if isinstance(tess_payload, dict) else None,
    }
    llm_context_text = "\n".join(part for part in [paddle_text, tesseract_text] if isinstance(part, str) and part.strip())
    return raw_ocr_context, llm_context_text


def _safe_text_len(value: Any) -> int:
    return len(str(value or "").strip())


def _ocr_rows_text_len(rows: Any) -> int:
    if not isinstance(rows, list):
        return 0
    total = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        total += _safe_text_len(row.get("row_text") or row.get("text"))
    return total


def _has_meaningful_ocr(extracted_json: Any, extracted_text: Any, ocr_debug: Any) -> bool:
    """Require real OCR signal before asking the LLM to invent structure."""
    if _safe_text_len(extracted_text) >= 20:
        return True

    if isinstance(extracted_json, dict):
        if _safe_text_len(extracted_json.get("text")) >= 20:
            return True
        if _ocr_rows_text_len(extracted_json.get("rows")) >= 20:
            return True

    if not isinstance(ocr_debug, dict):
        return False

    for source_key in ("paddle", "tesseract", "fused"):
        source = ocr_debug.get(source_key, {})
        if not isinstance(source, dict):
            continue
        if _safe_text_len(source.get("text")) >= 20:
            return True
        source_json = source.get("json")
        if isinstance(source_json, dict):
            if _safe_text_len(source_json.get("text")) >= 20:
                return True
            if _ocr_rows_text_len(source_json.get("rows")) >= 20:
                return True
    return False


def _ocr_error_summary(ocr_debug: Any) -> str:
    if not isinstance(ocr_debug, dict):
        return "no OCR debug payload"
    errors = ocr_debug.get("errors")
    if not isinstance(errors, dict) or not errors:
        return "no usable OCR text was extracted"
    return "; ".join(f"{key}: {value}" for key, value in errors.items())


def process_invoice_file(
    file_path: Path,
    invoice_type: str,
    out_json_path: Path,
    debug_json_path: Path,
    index: int,
    total: int,
) -> Dict[str, Any]:
    started = time.perf_counter()
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    debug_json_path.parent.mkdir(parents=True, exist_ok=True)
    ocr_debug: Dict[str, Any] = {}

    start_msg = f"[{index}/{total}] START FILE | type={invoice_type} | file={str(file_path)}"
    LOGGER.info(start_msg)
    print(start_msg, flush=True)
    if index == 1:
        warmup_msg = "NOTE: First file may take longer due OCR model warmup (Paddle init + cache load)."
        LOGGER.info(warmup_msg)
        print(warmup_msg, flush=True)

    try:
        ocr_started = time.perf_counter()
        _log_step(index, total, invoice_type, file_path, "OCR", "start")
        extracted_json, extracted_text, ocr_debug = run_multi_ocr(file_path)
        ocr_ms = int((time.perf_counter() - ocr_started) * 1000)
        paddle_rows = 0
        paddle_payload_dbg = ocr_debug.get("paddle", {}) if isinstance(ocr_debug, dict) else {}
        paddle_json_dbg = paddle_payload_dbg.get("json") if isinstance(paddle_payload_dbg, dict) else {}
        if isinstance(paddle_json_dbg, dict) and isinstance(paddle_json_dbg.get("rows"), list):
            paddle_rows = len(paddle_json_dbg.get("rows", []))

        if not _has_meaningful_ocr(extracted_json, extracted_text, ocr_debug):
            raise RuntimeError(
                "OCR failed: no usable text from PaddleOCR or Tesseract. "
                f"Details: {_ocr_error_summary(ocr_debug)}"
            )

        _log_step(
            index,
            total,
            invoice_type,
            file_path,
            "OCR",
            "done",
            extra=f"ocr_ms={ocr_ms} | fused_text_chars={len(extracted_text or '')} | paddle_rows={paddle_rows}",
        )

        pre_started = time.perf_counter()
        _log_step(index, total, invoice_type, file_path, "PREPROCESS", "start")
        print(f"[{index}/{total}] PREPROCESSING OCR output -> structured zones/fields...", flush=True)
        structured_data = preprocess_invoice(extracted_json=extracted_json, extracted_text=extracted_text)
        pre_ms = int((time.perf_counter() - pre_started) * 1000)
        if not isinstance(structured_data, dict):
            structured_data = {}
        products_count = len(structured_data.get("products", [])) if isinstance(structured_data.get("products"), list) else 0
        expenses_count = len(structured_data.get("expenses", [])) if isinstance(structured_data.get("expenses"), list) else 0
        _log_step(
            index,
            total,
            invoice_type,
            file_path,
            "PREPROCESS",
            "done",
            extra=f"pre_ms={pre_ms} | products={products_count} | expenses={expenses_count}",
        )

        paddle_payload = ocr_debug.get("paddle", {}) if isinstance(ocr_debug, dict) else {}
        paddle_json = paddle_payload.get("json") if isinstance(paddle_payload, dict) else {}
        if isinstance(paddle_json, dict):
            structured_data.setdefault("_ocr_rows", paddle_json.get("rows", []))
            structured_data.setdefault("_ocr_layout", paddle_json.get("layout", []))

        raw_ocr_context, llm_context_text = _prepare_llm_context(ocr_debug)
        prompt_text = load_prompt(invoice_type)

        llm_started = time.perf_counter()
        _log_step(index, total, invoice_type, file_path, "LLM", "start")
        print(f"[{index}/{total}] REFINING with LLM (remove ambiguities, map entities, enforce schema)...", flush=True)
        llm_output = refine_invoice_json_with_llm(
            structured_data=structured_data,
            extracted_text=llm_context_text,
            invoice_type=invoice_type,
            prompt_text=prompt_text,
            party_segmentation=None,
            raw_ocr_context=raw_ocr_context,
        )
        llm_ms = int((time.perf_counter() - llm_started) * 1000)

        if llm_output is None:
            final_json = normalize_final_invoice_json({}, structured_data, llm_context_text)
            llm_status = "fallback"
            print(f"[{index}/{total}] LLM returned fallback path, normalizing deterministic output...", flush=True)
        else:
            final_json = normalize_final_invoice_json(llm_output, structured_data, llm_context_text)
            llm_status = "success"
            print(f"[{index}/{total}] LLM refinement success, normalizing final JSON...", flush=True)
        _log_step(
            index,
            total,
            invoice_type,
            file_path,
            "LLM",
            "done",
            extra=f"llm={llm_status} | llm_ms={llm_ms}",
        )

        if not isinstance(final_json.get("productsArray"), list) or not final_json.get("productsArray"):
            final_json["productsArray"] = [
                {
                    "productName": p.get("productName"),
                    "Product_HSN_code": p.get("HSN"),
                    "Product_Quantity": p.get("quantity"),
                    "Product_Unit": p.get("unit"),
                    "Product_Rate": p.get("rate"),
                    "Product_Amount": p.get("amount"),
                }
                for p in structured_data.get("products", [])
                if isinstance(p, dict)
            ]

        save_started = time.perf_counter()
        _log_step(index, total, invoice_type, file_path, "SAVE", "start", extra=f"out={out_json_path}")
        print(f"[{index}/{total}] GENERATING FINAL JSON -> {str(out_json_path)}", flush=True)
        _safe_json_write(out_json_path, final_json)
        _safe_json_write(
            debug_json_path,
            {
                "invoice_type": invoice_type,
                "source_file": str(file_path),
                "ocr_debug": ocr_debug,
                "structured_data": structured_data,
                "llm_status": llm_status,
            },
        )
        save_ms = int((time.perf_counter() - save_started) * 1000)
        _log_step(
            index,
            total,
            invoice_type,
            file_path,
            "SAVE",
            "done",
            extra=f"save_ms={save_ms} | debug={debug_json_path}",
        )

        duration_ms = int((time.perf_counter() - started) * 1000)
        done_msg = f"[{index}/{total}] DONE FILE | type={invoice_type} | file={str(file_path)} | llm={llm_status} | total_ms={duration_ms}"
        LOGGER.info(done_msg)
        print(done_msg, flush=True)
        return {
            "status": "success",
            "file": str(file_path),
            "output_json": str(out_json_path),
            "debug_json": str(debug_json_path),
            "llm_status": llm_status,
            "duration_ms": duration_ms,
            "ocr_ms": ocr_ms,
            "preprocess_ms": pre_ms,
            "llm_ms": llm_ms,
            "save_ms": save_ms,
            "paddle_rows": paddle_rows,
            "fused_text_chars": len(extracted_text or ""),
            "admin_fields": _extract_admin_fields(final_json),
        }
    except Exception as exc:  # noqa: BLE001
        duration_ms = int((time.perf_counter() - started) * 1000)
        fail_msg = f"[{index}/{total}] FAILED FILE | type={invoice_type} | file={str(file_path)} | total_ms={duration_ms} | error={exc}"
        LOGGER.error(fail_msg, exc_info=LOGGER.isEnabledFor(logging.DEBUG))
        print(fail_msg, flush=True)
        error_payload = {
            "status": "error",
            "file": str(file_path),
            "output_json": None,
            "debug_json": None,
            "error": str(exc),
            "duration_ms": duration_ms,
            "ocr_debug": ocr_debug,
            "admin_fields": {},
        }
        error_debug_path = debug_json_path.with_name(f"{debug_json_path.stem}_error.json")
        error_payload["debug_json"] = str(error_debug_path)
        _safe_json_write(error_debug_path, error_payload)
        return error_payload


def _process_file(
    file_path: Path,
    invoice_type: str,
    output_dir: Path,
    debug_dir: Path,
    index: int,
    total: int,
) -> Dict[str, Any]:
    base = file_path.stem
    out_json_path = output_dir / f"{base}_final.json"
    debug_json_path = debug_dir / f"{base}_debug.json"
    return process_invoice_file(
        file_path=file_path,
        invoice_type=invoice_type,
        out_json_path=out_json_path,
        debug_json_path=debug_json_path,
        index=index,
        total=total,
    )


def _run_group(input_folder: Path, invoice_type: str, output_dir: Path, debug_dir: Path) -> Dict[str, Any]:
    files = _discover_files(input_folder)
    LOGGER.info("Group start | type=%s | folder=%s | files=%d", invoice_type, input_folder, len(files))
    print(f"GROUP START | type={invoice_type} | folder={str(input_folder)} | files={len(files)}", flush=True)
    for i, p in enumerate(files, start=1):
        print(f"  queued[{i}/{len(files)}]: {str(p)}", flush=True)
    group_started = time.perf_counter()
    results: List[Dict[str, Any]] = []
    for idx, path in enumerate(files, start=1):
        result = _process_file(path, invoice_type, output_dir, debug_dir, idx, len(files))
        results.append(result)
        success_so_far = sum(1 for r in results if r.get("status") == "success")
        elapsed = time.perf_counter() - group_started
        avg_per_file = elapsed / max(1, idx)
        remaining = int(avg_per_file * max(0, len(files) - idx))
        LOGGER.info(
            "Progress | type=%s | done=%d/%d | success=%d | failed=%d | eta_sec=%d",
            invoice_type,
            idx,
            len(files),
            success_so_far,
            idx - success_so_far,
            remaining,
        )
        print(
            f"PROGRESS | type={invoice_type} | done={idx}/{len(files)} | success={success_so_far} | failed={idx - success_so_far} | eta_sec={remaining}",
            flush=True,
        )
    success = sum(1 for r in results if r.get("status") == "success")
    failed = len(results) - success
    summary = {
        "invoice_type": invoice_type,
        "input_folder": str(input_folder),
        "output_folder": str(output_dir),
        "debug_folder": str(debug_dir),
        "total_files": len(files),
        "success": success,
        "failed": failed,
        "results": results,
    }
    LOGGER.info("Group done | type=%s | total=%d | success=%d | failed=%d", invoice_type, len(files), success, failed)
    print(f"GROUP DONE | type={invoice_type} | total={len(files)} | success={success} | failed={failed}", flush=True)
    return summary


def _flatten_admin_rows(final_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for section_key in ("purchase", "sales"):
        section = final_summary.get(section_key, {})
        if not isinstance(section, dict):
            continue
        invoice_type = section.get("invoice_type")
        for result in section.get("results", []):
            if not isinstance(result, dict):
                continue
            admin_fields = result.get("admin_fields", {}) if isinstance(result.get("admin_fields"), dict) else {}
            rows.append(
                {
                    "group": section_key,
                    "invoice_type": invoice_type,
                    "status": result.get("status"),
                    "file": result.get("file"),
                    "output_json": result.get("output_json"),
                    "debug_json": result.get("debug_json"),
                    "llm_status": result.get("llm_status"),
                    "duration_ms": result.get("duration_ms"),
                    "ocr_ms": result.get("ocr_ms"),
                    "preprocess_ms": result.get("preprocess_ms"),
                    "llm_ms": result.get("llm_ms"),
                    "save_ms": result.get("save_ms"),
                    "paddle_rows": result.get("paddle_rows"),
                    "fused_text_chars": result.get("fused_text_chars"),
                    "invoice_number": admin_fields.get("invoice_number"),
                    "invoice_date": admin_fields.get("invoice_date"),
                    "supplier_name": admin_fields.get("supplier_name"),
                    "customer_name": admin_fields.get("customer_name"),
                    "supplier_gst": admin_fields.get("supplier_gst"),
                    "customer_gst": admin_fields.get("customer_gst"),
                    "taxable_amount": admin_fields.get("taxable_amount"),
                    "total_amount": admin_fields.get("total_amount"),
                    "products_count": admin_fields.get("products_count"),
                    "expenses_count": admin_fields.get("expenses_count"),
                    "error": result.get("error"),
                }
            )
    return rows


def _write_admin_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "invoice_type",
        "status",
        "file",
        "output_json",
        "debug_json",
        "llm_status",
        "duration_ms",
        "ocr_ms",
        "preprocess_ms",
        "llm_ms",
        "save_ms",
        "paddle_rows",
        "fused_text_chars",
        "invoice_number",
        "invoice_date",
        "supplier_name",
        "customer_name",
        "supplier_gst",
        "customer_gst",
        "taxable_amount",
        "total_amount",
        "products_count",
        "expenses_count",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_admin_html(final_summary: Dict[str, Any], rows: List[Dict[str, Any]], generated_at: str) -> str:
    purchase = final_summary.get("purchase", {}) if isinstance(final_summary.get("purchase"), dict) else {}
    sales = final_summary.get("sales", {}) if isinstance(final_summary.get("sales"), dict) else {}
    total_files = len(rows)
    success = sum(1 for row in rows if row.get("status") == "success")
    failed = total_files - success
    fallback = sum(1 for row in rows if row.get("llm_status") == "fallback")
    success_llm = sum(1 for row in rows if row.get("llm_status") == "success")
    slowest = sorted(rows, key=lambda row: row.get("duration_ms") or 0, reverse=True)[:10]

    def _metric_card(label: str, value: Any) -> str:
        return (
            "<div class='card'>"
            f"<div class='label'>{html.escape(str(label))}</div>"
            f"<div class='value'>{html.escape(str(value))}</div>"
            "</div>"
        )

    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('invoice_type') or ''))}</td>"
            f"<td>{html.escape(str(Path(str(row.get('file') or '')).name))}</td>"
            f"<td>{html.escape(str(row.get('status') or ''))}</td>"
            f"<td>{html.escape(str(row.get('llm_status') or ''))}</td>"
            f"<td>{html.escape(str(row.get('invoice_number') or ''))}</td>"
            f"<td>{html.escape(str(row.get('supplier_name') or ''))}</td>"
            f"<td>{html.escape(str(row.get('customer_name') or ''))}</td>"
            f"<td>{html.escape(str(row.get('total_amount') or ''))}</td>"
            f"<td>{html.escape(str(row.get('duration_ms') or ''))}</td>"
            f"<td>{html.escape(str(row.get('output_json') or ''))}</td>"
            "</tr>"
        )

    slow_rows = []
    for row in slowest:
        slow_rows.append(
            "<tr>"
            f"<td>{html.escape(str(Path(str(row.get('file') or '')).name))}</td>"
            f"<td>{html.escape(str(row.get('invoice_type') or ''))}</td>"
            f"<td>{html.escape(str(row.get('duration_ms') or ''))}</td>"
            f"<td>{html.escape(str(row.get('ocr_ms') or ''))}</td>"
            f"<td>{html.escape(str(row.get('llm_ms') or ''))}</td>"
            "</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Admin Monitoring Dashboard</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; background: #f6f8fb; color: #1f2937; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .muted {{ color: #6b7280; margin-bottom: 20px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; margin: 18px 0 28px; }}
    .card {{ background: white; border-radius: 12px; padding: 16px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); }}
    .label {{ font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: .04em; }}
    .value {{ font-size: 28px; font-weight: 700; margin-top: 8px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; margin-bottom: 24px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 13px; }}
    th {{ background: #111827; color: white; position: sticky; top: 0; }}
    tr:nth-child(even) td {{ background: #f9fafb; }}
    .section {{ margin-top: 28px; }}
  </style>
</head>
<body>
  <h1>Admin Monitoring Dashboard</h1>
  <div class="muted">Generated at: {html.escape(generated_at)}</div>

  <div class="grid">
    {_metric_card("Total Files", total_files)}
    {_metric_card("Success", success)}
    {_metric_card("Failed", failed)}
    {_metric_card("LLM Fallback", fallback)}
    {_metric_card("LLM Success", success_llm)}
    {_metric_card("Purchase Files", purchase.get("total_files", 0))}
    {_metric_card("Sales Files", sales.get("total_files", 0))}
    {_metric_card("Run Duration (ms)", final_summary.get("duration_ms", 0))}
  </div>

  <div class="section">
    <h2>Processed Files</h2>
    <table>
      <thead>
        <tr>
          <th>Invoice Type</th>
          <th>File</th>
          <th>Status</th>
          <th>LLM</th>
          <th>Invoice No</th>
          <th>Supplier</th>
          <th>Customer</th>
          <th>Total Amount</th>
          <th>Duration (ms)</th>
          <th>Output JSON</th>
        </tr>
      </thead>
      <tbody>
        {''.join(table_rows)}
      </tbody>
    </table>
  </div>

  <div class="section">
    <h2>Slowest Files</h2>
    <table>
      <thead>
        <tr>
          <th>File</th>
          <th>Invoice Type</th>
          <th>Total (ms)</th>
          <th>OCR (ms)</th>
          <th>LLM (ms)</th>
        </tr>
      </thead>
      <tbody>
        {''.join(slow_rows)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""


def _write_admin_monitoring(root: Path, admin_dir: Path, final_summary: Dict[str, Any]) -> Dict[str, str]:
    admin_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = admin_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = _flatten_admin_rows(final_summary)

    summary_json = run_dir / "summary.json"
    manifest_csv = run_dir / "processed_files.csv"
    dashboard_html = run_dir / "dashboard.html"
    latest_json = admin_dir / "latest_summary.json"
    latest_csv = admin_dir / "latest_processed_files.csv"
    latest_html = admin_dir / "latest_dashboard.html"

    _safe_json_write(summary_json, final_summary)
    _write_admin_csv(manifest_csv, rows)
    html_text = _render_admin_html(final_summary, rows, generated_at=run_id)
    _safe_text_write(dashboard_html, html_text)

    _safe_json_write(latest_json, final_summary)
    _write_admin_csv(latest_csv, rows)
    _safe_text_write(latest_html, html_text)

    return {
        "run_dir": str(run_dir),
        "summary_json": str(summary_json),
        "manifest_csv": str(manifest_csv),
        "dashboard_html": str(dashboard_html),
        "latest_summary_json": str(latest_json),
        "latest_manifest_csv": str(latest_csv),
        "latest_dashboard_html": str(latest_html),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automatic folder-driven invoice batch processing pipeline")
    parser.add_argument("--purchase-dir", default="PURCHASE", help="Input folder for purchase invoices")
    parser.add_argument("--sales-dir", default="SALES", help="Input folder for sales invoices")
    parser.add_argument("--sales-dir-fallback", default="SALE", help="Fallback sales input folder if SALES does not exist")
    parser.add_argument("--purchase-out", default="PURCHASE_JSON", help="Output folder for purchase JSONs (root-level)")
    parser.add_argument("--purchase-debug-out", default="PURCHASE_DEBUG", help="Debug folder for purchase artifacts (root-level)")
    parser.add_argument("--sales-out", default="SALES_JSON", help="Output folder for sales JSONs (root-level)")
    parser.add_argument("--sales-debug-out", default="SALES_DEBUG", help="Debug folder for sales artifacts (root-level)")
    parser.add_argument("--admin-out", default="ADMIN_MONITORING", help="Admin monitoring folder for summaries, CSVs, and dashboard")
    parser.add_argument("--summary-file", default="batch_summary.json", help="Summary JSON path (root-level)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Terminal log level")
    parser.add_argument("--log-file", default=None, help="Optional log file path in project root")
    return parser.parse_args(argv)


def _setup_logging(log_level: str, log_file: Optional[str]) -> None:
    level = getattr(logging, str(log_level).upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)

    # Keep terminal output focused on our explicit print() progress lines.
    # Console logging is reserved for warnings/errors from libraries.
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter(fmt))
    root.addHandler(console)

    if isinstance(log_file, str) and log_file.strip():
        log_path = Path(__file__).resolve().parent / log_file.strip()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(fmt))
        root.addHandler(file_handler)
        LOGGER.info("Logging to file: %s", log_path)

    logging.getLogger("pytesseract").setLevel(logging.WARNING)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _setup_logging(args.log_level, args.log_file)

    root = Path(__file__).resolve().parent
    purchase_in = root / args.purchase_dir
    sales_in = root / args.sales_dir
    if not sales_in.exists():
        fallback = root / args.sales_dir_fallback
        if fallback.exists():
            sales_in = fallback

    purchase_out = root / args.purchase_out
    purchase_debug_out = root / args.purchase_debug_out
    sales_out = root / args.sales_out
    sales_debug_out = root / args.sales_debug_out
    admin_out = root / args.admin_out

    started = time.perf_counter()
    purchase_summary = _run_group(purchase_in, "Purchase Invoice", purchase_out, purchase_debug_out)
    sales_summary = _run_group(sales_in, "Sales Invoice", sales_out, sales_debug_out)
    total_ms = int((time.perf_counter() - started) * 1000)

    final_summary = {
        "status": "completed",
        "root": str(root),
        "duration_ms": total_ms,
        "purchase": purchase_summary,
        "sales": sales_summary,
    }
    _safe_json_write(root / args.summary_file, final_summary)
    admin_artifacts = _write_admin_monitoring(root, admin_out, final_summary)
    final_summary["admin_monitoring"] = admin_artifacts
    _safe_json_write(root / args.summary_file, final_summary)
    LOGGER.info("Batch complete | summary=%s | duration_ms=%d", root / args.summary_file, total_ms)
    print(f"ADMIN DASHBOARD | {admin_artifacts['latest_dashboard_html']}", flush=True)
    print(f"ADMIN CSV | {admin_artifacts['latest_manifest_csv']}", flush=True)
    print(f"BATCH COMPLETE | summary={str(root / args.summary_file)} | duration_ms={total_ms}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
