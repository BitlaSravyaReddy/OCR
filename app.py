import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import logging

import streamlit as st

from llm.refine import normalize_final_invoice_json, refine_invoice_json_with_llm
from ocr.ocr_pipeline import run_multi_ocr, supported_upload_extensions


st.set_page_config(page_title="Invoice Extraction System", layout="wide")

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
else:
    logging.getLogger().setLevel(logging.INFO)

LOGGER = logging.getLogger(__name__)
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _extract_summary_fields(final_json: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Any]]:
    invoice_number = final_json.get("Invoice_Number")
    total_amount = final_json.get("Total_Amount")
    return invoice_number, total_amount


TOP_STRING_FIELDS = {
    "Customer_Name",
    "SupplierName",
    "Customer_address",
    "Supplier_address",
    "Customer_GSTIN",
    "Supplier_GST",
    "Invoice_Number",
    "Invoice_Date",
    "Vehicle_Number",
    "Bank_Name",
    "bank_account_number",
    "IFSCCode",
    "Email",
    "Phone",
    "Supplier_First_Word",
}

TOP_NUMERIC_FIELDS = {
    "SGST_Amount",
    "CGST_Amount",
    "IGST_Amount",
    "Total_Expenses",
    "Taxable_Amount",
    "Total_Amount",
}

PRODUCT_STRING_FIELDS = {
    "productName",
    "Product_HSN_code",
    "Product_Unit",
    "Product_Description",
    "Product_BatchNo",
    "Product_ExpDate",
    "Product_MfgDate",
}

PRODUCT_NUMERIC_FIELDS = {
    "Product_GST_Rate",
    "Product_Quantity",
    "Product_Rate",
    "Product_DisPer",
    "Product_Amount",
    "Product_MRP",
}

EXPENSE_STRING_FIELDS = {"Expense_Name"}
EXPENSE_NUMERIC_FIELDS = {"Expense_Percentage", "Expense_Amount"}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _audit_datatypes(final_json: Dict[str, Any]) -> Dict[str, Any]:
    violations: list[dict[str, Any]] = []

    for key in TOP_STRING_FIELDS:
        value = final_json.get(key)
        if not isinstance(value, str):
            violations.append({"field": key, "expected": "string", "actual_type": type(value).__name__, "value": value})

    for key in TOP_NUMERIC_FIELDS:
        value = final_json.get(key)
        if not _is_number(value):
            violations.append({"field": key, "expected": "number", "actual_type": type(value).__name__, "value": value})

    products = final_json.get("productsArray")
    if not isinstance(products, list):
        violations.append({"field": "productsArray", "expected": "list", "actual_type": type(products).__name__, "value": products})
        products = []
    for idx, item in enumerate(products):
        if not isinstance(item, dict):
            violations.append({"field": f"productsArray[{idx}]", "expected": "object", "actual_type": type(item).__name__, "value": item})
            continue
        for key in PRODUCT_STRING_FIELDS:
            value = item.get(key)
            if not isinstance(value, str):
                violations.append(
                    {"field": f"productsArray[{idx}].{key}", "expected": "string", "actual_type": type(value).__name__, "value": value}
                )
        for key in PRODUCT_NUMERIC_FIELDS:
            value = item.get(key)
            if not _is_number(value):
                violations.append(
                    {"field": f"productsArray[{idx}].{key}", "expected": "number", "actual_type": type(value).__name__, "value": value}
                )

    expenses = final_json.get("Expenses")
    if not isinstance(expenses, list):
        violations.append({"field": "Expenses", "expected": "list", "actual_type": type(expenses).__name__, "value": expenses})
        expenses = []
    for idx, item in enumerate(expenses):
        if not isinstance(item, dict):
            violations.append({"field": f"Expenses[{idx}]", "expected": "object", "actual_type": type(item).__name__, "value": item})
            continue
        for key in EXPENSE_STRING_FIELDS:
            value = item.get(key)
            if not isinstance(value, str):
                violations.append(
                    {"field": f"Expenses[{idx}].{key}", "expected": "string", "actual_type": type(value).__name__, "value": value}
                )
        for key in EXPENSE_NUMERIC_FIELDS:
            value = item.get(key)
            if not _is_number(value):
                violations.append(
                    {"field": f"Expenses[{idx}].{key}", "expected": "number", "actual_type": type(value).__name__, "value": value}
                )

    return {
        "ok": len(violations) == 0,
        "violation_count": len(violations),
        "violations": violations,
    }


def _load_prompt_from_prompts_dir(invoice_type: str) -> str:
    file_name = "Purchase_Invoice.txt" if invoice_type == "Purchase Invoice" else "Sales_Invoice.txt"
    prompt_file = PROMPTS_DIR / file_name
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    LOGGER.info("app.py using prompt file: %s", prompt_file)
    return prompt_file.read_text(encoding="utf-8")


def _render_debug_downloads(uploaded_name: str, ocr_debug: Dict[str, Any]) -> None:
    base_name = Path(uploaded_name).stem
    paddle_payload = ocr_debug.get("paddle", {}) if isinstance(ocr_debug, dict) else {}
    tesseract_payload = ocr_debug.get("tesseract", {}) if isinstance(ocr_debug, dict) else {}
    fused_payload = ocr_debug.get("fused", {}) if isinstance(ocr_debug, dict) else {}
    errors_payload = ocr_debug.get("errors", {}) if isinstance(ocr_debug, dict) else {}

    st.download_button(
        label="Download Paddle OCR JSON",
        data=json.dumps(paddle_payload.get("json", {}), ensure_ascii=False, indent=2),
        file_name=f"{base_name}_paddle.json",
        mime="application/json",
    )
    st.download_button(
        label="Download Paddle OCR Text",
        data=str(paddle_payload.get("text", "") or ""),
        file_name=f"{base_name}_paddle.txt",
        mime="text/plain",
    )
    st.download_button(
        label="Download Tesseract OCR JSON",
        data=json.dumps(
            {
                "text": tesseract_payload.get("text", ""),
                "boxes": tesseract_payload.get("boxes"),
                "text_length": tesseract_payload.get("text_length"),
                "boxes_count": tesseract_payload.get("boxes_count"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        file_name=f"{base_name}_tesseract.json",
        mime="application/json",
    )
    st.download_button(
        label="Download Fused OCR JSON",
        data=json.dumps(fused_payload.get("json", {}), ensure_ascii=False, indent=2),
        file_name=f"{base_name}_fused.json",
        mime="application/json",
    )
    st.download_button(
        label="Download Fused OCR Text",
        data=str(fused_payload.get("text", "") or ""),
        file_name=f"{base_name}_fused.txt",
        mime="text/plain",
    )
    st.download_button(
        label="Download OCR Errors JSON",
        data=json.dumps(errors_payload, ensure_ascii=False, indent=2),
        file_name=f"{base_name}_ocr_errors.json",
        mime="application/json",
    )


@st.cache_data(show_spinner=False)
def _cached_multi_ocr(
    file_bytes: bytes,
    suffix: str,
    cache_version: str = "multi_ocr_v4",
) -> Tuple[Dict[str, Any], str, Dict[str, Any]]:
    """
    Cached OCR pass for repeated uploads of the same file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        path = Path(tmp.name)
    try:
        fused_json, fused_text, debug_payload = run_multi_ocr(path)
        return fused_json, fused_text, debug_payload
    finally:
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass


def main() -> None:
    st.title("Invoice Extraction System")
    if "last_run" not in st.session_state:
        st.session_state["last_run"] = None

    with st.container():
        uploaded_file = st.file_uploader(
            "Upload Invoice (PDF/Image)",
            type=supported_upload_extensions(),
            accept_multiple_files=False,
        )
        invoice_type = st.selectbox(
            "Select Invoice Type",
            options=["Purchase Invoice", "Sales Invoice"],
            index=0,
        )
        show_debug = st.toggle("Show intermediate steps", value=False)

    if st.button("Process Invoice", type="primary"):
        if uploaded_file is None:
            st.error("Please upload a PDF or image invoice before processing.")
            return

        step_status = st.empty()
        with st.spinner("Processing invoice..."):
            try:
                file_bytes = uploaded_file.getbuffer().tobytes()
                suffix = Path(uploaded_file.name).suffix or ".bin"

                step_status.info("Running OCR...")
                extracted_json, extracted_text, ocr_debug = _cached_multi_ocr(file_bytes, suffix, "multi_ocr_v4")
                if not extracted_json and not extracted_text:
                    st.error("OCR failed: no text/data could be extracted from the file.")
                    return

                paddle_payload = ocr_debug.get("paddle", {}) if isinstance(ocr_debug, dict) else {}
                paddle_json = paddle_payload.get("json") if isinstance(paddle_payload, dict) else None
                paddle_text = paddle_payload.get("text") if isinstance(paddle_payload, dict) else ""
                tesseract_payload = ocr_debug.get("tesseract", {}) if isinstance(ocr_debug, dict) else {}
                tesseract_text = tesseract_payload.get("text") if isinstance(tesseract_payload, dict) else ""
                # Do not use preprocessing/fusion-derived structure for LLM guidance.
                # LLM must infer directly from raw OCR engine outputs.
                structured_data = {
                    "header": {},
                    "parties": {},
                    "products": [],
                    "expenses": [],
                    "totals": {},
                    "_ocr_rows": paddle_json.get("rows", []) if isinstance(paddle_json, dict) else [],
                    "_ocr_layout": paddle_json.get("layout", []) if isinstance(paddle_json, dict) else [],
                }

                # Pass raw OCR engine outputs to LLM for better entity resolution.
                raw_ocr_context = {
                    "paddle_json": paddle_json,
                    "paddle_text": paddle_text,
                    "tesseract_text": tesseract_text,
                    "tesseract_boxes": tesseract_payload.get("boxes") if isinstance(tesseract_payload, dict) else None,
                }
                llm_context_text = "\n".join(part for part in [paddle_text, tesseract_text] if part)

                step_status.info("Generating final JSON from raw OCR...")
                prompt_text = _load_prompt_from_prompts_dir(invoice_type)
                llm_output = refine_invoice_json_with_llm(
                    structured_data=structured_data,
                    extracted_text=llm_context_text,
                    invoice_type=invoice_type,
                    prompt_text=prompt_text,
                    party_segmentation=None,
                    raw_ocr_context=raw_ocr_context,
                )

                if llm_output is None:
                    st.warning(
                        "LLM refinement returned non-JSON response. "
                        "Showing structured preprocessed JSON fallback (check terminal logs for LLM parse details)."
                    )
                    final_json = normalize_final_invoice_json({}, structured_data, llm_context_text)
                else:
                    final_json = normalize_final_invoice_json(llm_output, structured_data, llm_context_text)

                # Final UI guardrail: every invoice must have product details.
                if not isinstance(final_json.get("productsArray"), list) or len(final_json.get("productsArray", [])) == 0:
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
                if not final_json.get("productsArray"):
                    st.error("Product details are mandatory but could not be extracted. Please review OCR quality.")
                    return

                st.session_state["last_run"] = {
                    "uploaded_name": uploaded_file.name,
                    "final_json": final_json,
                    "ocr_debug": ocr_debug,
                    "structured_data": structured_data,
                    "show_debug": show_debug,
                }

                step_status.success("Processing complete.")

            except FileNotFoundError as exc:
                st.error(f"Prompt loading failed: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Processing failed: {exc}")

    last_run = st.session_state.get("last_run")
    if isinstance(last_run, dict):
        final_json = last_run.get("final_json", {})
        ocr_debug = last_run.get("ocr_debug", {})
        structured_data = last_run.get("structured_data", {})
        uploaded_name = str(last_run.get("uploaded_name") or "invoice")
        debug_enabled = bool(last_run.get("show_debug", False))

        invoice_number, total_amount = _extract_summary_fields(final_json)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Invoice Number", str(invoice_number) if invoice_number is not None else "N/A")
        with c2:
            st.metric("Total Amount", str(total_amount) if total_amount is not None else "N/A")

        st.subheader("Final Structured JSON")
        st.json(final_json)

        audit = _audit_datatypes(final_json)
        with st.expander("Datatype Audit", expanded=not audit["ok"]):
            if audit["ok"]:
                st.success("Datatype contract check passed.")
            else:
                st.warning(f"Datatype violations found: {audit['violation_count']}")
                st.json(audit["violations"])

        st.download_button(
            label="Download JSON",
            data=json.dumps(final_json, ensure_ascii=False, indent=2),
            file_name=f"{Path(uploaded_name).stem}_final.json",
            mime="application/json",
        )
        with st.expander("Download OCR Artifacts", expanded=False):
            _render_debug_downloads(uploaded_name, ocr_debug)

        if debug_enabled:
            with st.expander("Paddle OCR Output", expanded=False):
                st.json(ocr_debug.get("paddle", {}))
            with st.expander("Tesseract OCR Output", expanded=False):
                st.json(ocr_debug.get("tesseract", {}))
            with st.expander("Fused OCR Output", expanded=False):
                st.json(ocr_debug.get("fused", {}))
            with st.expander("OCR Errors", expanded=False):
                st.json(ocr_debug.get("errors", {}))
            with st.expander("Structured Intermediate JSON", expanded=False):
                st.json(structured_data)


if __name__ == "__main__":
    main()
