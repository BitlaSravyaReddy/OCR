import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import re
import ast
import os

from google import genai
from openai import OpenAI

from llm.embedded_prompts import PURCHASE_INVOICE_PROMPT, SALES_INVOICE_PROMPT
from llm_request import load_api_key

LOGGER = logging.getLogger(__name__)

REMOVE_LINES_IF_CONTAIN = (
    "original",
    "duplicate",
    "triplicate",
    "transport",
    "remarks",
    "reference",
)

LLM_NOISE_LINE_PATTERNS = (
    re.compile(r"\birn\b", re.I),
    re.compile(r"\back\s*no\b", re.I),
    re.compile(r"\bmsme\b", re.I),
    re.compile(r"\budyam\b", re.I),
    re.compile(r"\bcin\b", re.I),
    re.compile(r"\bqr\s*code\b", re.I),
    re.compile(r"\bdeclaration\b", re.I),
    re.compile(r"\bterms\s*&?\s*condition", re.I),
    re.compile(r"\bthis is a computer generated\b", re.I),
)

EXPENSE_BANNED_TOKENS = (
    "prev due balance",
    "a/c no",
    "ifsc",
    " dr ",
    "dr.",
    " dr:",
)


def load_prompt(invoice_type: str) -> str:
    """
    Load invoice prompt from local prompts directory by invoice type.
    """
    prompts_dir = Path(__file__).resolve().parents[1] / "prompts"
    file_name = "Purchase_Invoice.txt" if invoice_type == "Purchase Invoice" else "Sales_Invoice.txt"
    prompt_file = prompts_dir / file_name
    if prompt_file.exists():
        LOGGER.info("Using prompt file: %s", prompt_file)
        return prompt_file.read_text(encoding="utf-8")
    LOGGER.info("Prompt file not found (%s). Using embedded prompt fallback.", prompt_file)
    return PURCHASE_INVOICE_PROMPT if invoice_type == "Purchase Invoice" else SALES_INVOICE_PROMPT


def _trim_for_prompt(value: Any, max_rows: int = 40) -> Any:
    if not isinstance(value, dict):
        return value
    out = dict(value)
    # Do not pass noisy OCR-derived normalized/record fields to LLM.
    out.pop("normalized_fields", None)
    out.pop("records", None)
    rows = out.get("rows")
    if isinstance(rows, list) and len(rows) > max_rows:
        out["rows"] = rows[:max_rows]
    layout = out.get("layout")
    if isinstance(layout, list) and len(layout) > max_rows:
        out["layout"] = layout[:max_rows]
    # Hard filter: remove noisy OCR rows before LLM guidance.
    rows_filtered: List[Dict[str, Any]] = []
    for row in out.get("rows", []) if isinstance(out.get("rows"), list) else []:
        if not isinstance(row, dict):
            continue
        rt = str(row.get("row_text", "") or "")
        low = rt.lower()
        if any(tok in low for tok in REMOVE_LINES_IF_CONTAIN):
            continue
        if any(p.search(rt) for p in LLM_NOISE_LINE_PATTERNS):
            continue
        if re.search(r"\b[a-f0-9]{10,}\b", rt, re.I):
            continue
        if re.fullmatch(r"\s*\d{8,}\s*", rt):
            continue
        rows_filtered.append(row)
    if rows_filtered:
        out["rows"] = rows_filtered[:max_rows]
    return out


def _trim_boxes_for_prompt(value: Any, limit: int = 100) -> Any:
    if not isinstance(value, list):
        return value
    return value[:limit]


def _strip_noise_lines(text: str) -> str:
    if not text:
        return ""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    filtered = []
    for ln in lines:
        low = ln.lower()
        if any(tok in low for tok in REMOVE_LINES_IF_CONTAIN):
            continue
        if any(p.search(ln) for p in LLM_NOISE_LINE_PATTERNS):
            continue
        if re.search(r"\b[a-f0-9]{10,}\b", ln, re.I):
            continue
        if re.fullmatch(r"\s*\d{8,}\s*", ln):
            continue
        filtered.append(ln)
    return "\n".join(filtered)


def _has_meaningful_structured_data(data: Dict[str, Any]) -> bool:
    if not isinstance(data, dict):
        return False
    for key in ("header", "parties", "totals"):
        section = data.get(key)
        if isinstance(section, dict) and any(v not in (None, "", [], {}, "null") for v in section.values()):
            return True
    for key in ("products", "expenses"):
        section = data.get(key)
        if isinstance(section, list) and len(section) > 0:
            return True
    return False


def refine_invoice_json_with_llm(
    structured_data: Dict[str, Any],
    extracted_text: str,
    invoice_type: str,
    model_name: str = "gemini-2.5-flash",
    openai_model: Optional[str] = None,
    prompt_text: Optional[str] = None,
    party_segmentation: Optional[Dict[str, Any]] = None,
    raw_ocr_context: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Refine preprocessed intermediate JSON using Gemini.
    """
    prompt = prompt_text if prompt_text is not None else load_prompt(invoice_type)
    cleaned_extracted_text = _strip_noise_lines(extracted_text or "")

    if model_name.startswith("models/"):
        model_name = model_name[7:]

    has_structured_signal = _has_meaningful_structured_data(structured_data)

    message = f"""You are a strict JSON generator for invoice refinement from OCR outputs.
Output policy (must follow):
- Output exactly one JSON object.
- First character must be '{{' and last character must be '}}'.
- No markdown fences, no prose, no comments.
- Use null for unknown values.

{prompt}

# Preprocessed Structured Data (primary source):
```json
{json.dumps(structured_data, ensure_ascii=False, indent=2)}
```

# OCR Raw Text (fallback context only):
```
{cleaned_extracted_text[:3000]}
```

Rules:
1) Input is OCR text + OCR-structured JSON, not an image. Infer entities from OCR context only.
2) Use preprocessed structured data as the primary source.
3) Use OCR raw text only to fill missing values.
4) Ignore OCR `normalized_fields` and `records`; they are noisy. Trust rows/layout/raw text evidence.
5) Identify party entities carefully: company/person names vs address lines must not be mixed.
6) Preserve monetary values and line items.
7) Return ONLY valid JSON.
8) Preserve all product line items from preprocessed data; do not collapse or summarize them.
9) Keep product quantity/rate/amount/HSN as extracted unless OCR fallback clearly corrects them.
10) Invoice_Number must NOT be a date. Reject date-like tokens (e.g., 14/03/2026, 3-Nov-2025) for Invoice_Number.
11) Address fields must contain only postal address text. Exclude logistics/metadata labels like delivery note, despatch, truck, terms, references.
"""
    if not has_structured_signal:
        message += """

Primary source policy:
- Preprocessed structure is intentionally minimal for this run.
- Use raw OCR engine evidence (Paddle/Tesseract) as the primary source of truth.
"""
    if isinstance(raw_ocr_context, dict):
        paddle_json = raw_ocr_context.get("paddle_json")
        paddle_text = _strip_noise_lines(raw_ocr_context.get("paddle_text") or "")
        tesseract_text = _strip_noise_lines(raw_ocr_context.get("tesseract_text") or "")
        tesseract_boxes = raw_ocr_context.get("tesseract_boxes")
        message += f"""

# Raw OCR Engine Evidence (entity validation):
Paddle OCR JSON (trimmed):
```json
{json.dumps(_trim_for_prompt(paddle_json, max_rows=40), ensure_ascii=False, indent=2)}
```

Paddle OCR text (trimmed):
```
{(paddle_text or "")[:1800]}
```

Tesseract OCR text (trimmed):
```
{(tesseract_text or "")[:1800]}
```

Tesseract boxes sample (trimmed):
```json
{json.dumps(_trim_boxes_for_prompt(tesseract_boxes, limit=100), ensure_ascii=False, indent=2)}
```
"""
    parties = structured_data.get("parties", {}) if isinstance(structured_data.get("parties"), dict) else {}
    supplier_candidates = parties.get("_supplier_candidates", []) if isinstance(parties.get("_supplier_candidates"), list) else []
    customer_candidates = parties.get("_customer_candidates", []) if isinstance(parties.get("_customer_candidates"), list) else []
    if supplier_candidates or customer_candidates:
        compact = {
            "supplier_candidates": supplier_candidates[:3],
            "customer_candidates": customer_candidates[:3],
            "supplier_gst": parties.get("Supplier_GST"),
            "customer_gst": parties.get("Customer_GSTIN"),
        }
        message += f"""

# Compact Party Evidence (highest priority for names):
```json
{json.dumps(compact, ensure_ascii=False, indent=2)}
```
"""
    structure_debug = structured_data.get("_structure_debug", {}) if isinstance(structured_data, dict) else {}
    zones = structure_debug.get("zones", {}) if isinstance(structure_debug, dict) else {}
    if isinstance(zones, dict) and zones:
        compact_zones = {
            "supplier_block": (zones.get("supplier_block") or [])[:10],
            "buyer_block": (zones.get("buyer_block") or [])[:10],
            "invoice_meta": (zones.get("invoice_meta") or [])[:10],
            "items_block": (zones.get("items_block") or [])[:20],
            "totals_block": (zones.get("totals_block") or [])[:20],
        }
        message += f"""

# Structured Zones (use these sections first, avoid cross-zone mixing):
```json
{json.dumps(compact_zones, ensure_ascii=False, indent=2)}
```
"""
    if party_segmentation:
        message += f"""

# Identified Parties (ownership guidance):
```json
{json.dumps(party_segmentation, ensure_ascii=False, indent=2)}
```

Party Rules:
1) Supplier and Customer are strictly different entities.
2) Do NOT copy GST/name fields across parties.
3) If a party field is uncertain, keep it null.
4) Customer_GSTIN must be null if customer GST is not explicitly present in invoice.
5) Never copy Supplier_GST into Customer_GSTIN.
"""

    expected_products = 0
    expected_expenses = 0
    if isinstance(structured_data.get("products"), list):
        expected_products = len(structured_data.get("products", []))
    if isinstance(structured_data.get("expenses"), list):
        expected_expenses = len(structured_data.get("expenses", []))

    message += f"""

Consistency requirements:
- productsArray should retain line-item granularity from preprocessed products.
- Expected minimum productsArray count: {expected_products}
- Expected minimum Expenses count: {expected_expenses}
- Include `_meta` object and `tags` array in output.
"""

    config = {
        "temperature": 0.0,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 4096,
        "response_mime_type": "application/json",
        "response_schema": {
            "type": "OBJECT",
            "properties": {
                "Customer_Name": {"type": "STRING", "nullable": True},
                "SupplierName": {"type": "STRING", "nullable": True},
                "Customer_GSTIN": {"type": "STRING", "nullable": True},
                "Supplier_GST": {"type": "STRING", "nullable": True},
                "Invoice_Number": {"type": "STRING", "nullable": True},
                "Invoice_Date": {"type": "STRING", "nullable": True},
                "Taxable_Amount": {"type": "NUMBER", "nullable": True},
                "Total_Amount": {"type": "NUMBER", "nullable": True},
                "productsArray": {
                    "type": "ARRAY",
                    "items": {"type": "OBJECT"},
                },
                "Expenses": {
                    "type": "ARRAY",
                    "items": {"type": "OBJECT"},
                },
                "_meta": {"type": "OBJECT", "nullable": True},
                "tags": {"type": "ARRAY", "items": {"type": "STRING"}, "nullable": True},
            },
            "required": ["productsArray", "Expenses", "_meta", "tags"],
        },
    }

    openai_api_key = _load_openai_api_key()
    gemini_api_key = load_api_key()

    # Primary provider: OpenAI
    if openai_api_key:
        selected_openai_model = openai_model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        try:
            parsed = _run_openai_refinement(
                api_key=openai_api_key,
                model=selected_openai_model,
                message=message,
            )
            if parsed is not None:
                LOGGER.info("LLM provider used: OpenAI | model=%s", selected_openai_model)
                return _merge_with_fallback(parsed, structured_data, party_segmentation, extracted_text)
            LOGGER.warning("OpenAI refinement returned non-JSON output; trying Gemini fallback.")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("OpenAI refinement failed; trying Gemini fallback | error=%s", exc)

    # Fallback provider: Gemini
    if gemini_api_key:
        try:
            parsed = _run_gemini_refinement(
                api_key=gemini_api_key,
                model_name=model_name,
                message=message,
                config=config,
            )
            if parsed is not None:
                LOGGER.info("LLM provider used: Gemini | model=%s", model_name)
                return _merge_with_fallback(parsed, structured_data, party_segmentation, extracted_text)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Gemini refinement failed | error=%s", exc)
            return None

    raise ValueError("Missing LLM API keys. Set OPENAI_API_KEY (preferred) or GEMINI_API_KEY/GOOGLE_API_KEY.")


def _run_openai_refinement(api_key: str, model: str, message: str) -> Optional[Dict[str, Any]]:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a strict JSON generator. Output exactly one valid JSON object and nothing else.",
            },
            {"role": "user", "content": message},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    text = ""
    if response.choices and response.choices[0].message:
        text = response.choices[0].message.content or ""

    parsed = _parse_json(text)
    if parsed is not None:
        return parsed

    retry = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Return only one valid JSON object with double-quoted keys. No markdown.",
            },
            {"role": "user", "content": message},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    retry_text = ""
    if retry.choices and retry.choices[0].message:
        retry_text = retry.choices[0].message.content or ""
    return _parse_json(retry_text)


def _run_gemini_refinement(
    api_key: str,
    model_name: str,
    message: str,
    config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(model=model_name, contents=message, config=config)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("LLM schema config not accepted, retrying without schema | error=%s", exc)
        fallback_config = dict(config)
        fallback_config.pop("response_schema", None)
        response = client.models.generate_content(model=model_name, contents=message, config=fallback_config)
    response_text = _extract_response_text(response)
    parsed = _parse_json(response_text)
    if parsed is not None:
        LOGGER.info("LLM refinement parse success | chars=%s", len(response_text or ""))
        return parsed

    LOGGER.warning(
        "LLM refinement parse failed on first pass | chars=%s | preview=%s",
        len(response_text or ""),
        (response_text or "")[:200].replace("\n", " "),
    )

    retry_message = (
        "Return only one valid JSON object with double-quoted keys and values where needed. "
        "Do not include markdown, explanation, or comments.\n\n"
        + message
    )
    try:
        retry_response = client.models.generate_content(model=model_name, contents=retry_message, config=config)
    except Exception:
        fallback_config = dict(config)
        fallback_config.pop("response_schema", None)
        retry_response = client.models.generate_content(model=model_name, contents=retry_message, config=fallback_config)
    retry_text = _extract_response_text(retry_response)
    retry_parsed = _parse_json(retry_text)
    if retry_parsed is None:
        LOGGER.error(
            "LLM refinement parse failed after retry | chars=%s | preview=%s",
            len(retry_text or ""),
            (retry_text or "")[:200].replace("\n", " "),
        )
        return None
    LOGGER.info("LLM refinement parse success after retry | chars=%s", len(retry_text or ""))
    return retry_parsed


def _load_openai_api_key() -> str:
    env_val = os.getenv("OPENAI_API_KEY") or os.getenv("openai_API_KEY") or os.getenv("OPENAI_APIKEY")
    if env_val:
        return env_val.strip().strip('"').strip("'")

    env_path = Path(".env")
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() in {"OPENAI_API_KEY", "openai_API_KEY", "OPENAI_APIKEY"}:
                return value.strip().strip('"').strip("'")
    return ""


def _parse_json(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    for fence in ("```json", "```"):
        if fence in text:
            start = text.find(fence) + len(fence)
            end = text.find("```", start)
            if end > start:
                snippet = text[start:end].strip()
                try:
                    value = json.loads(snippet)
                    return value if isinstance(value, dict) else None
                except json.JSONDecodeError:
                    pass

    snippet = _extract_balanced_json_object(text)
    if snippet:
        value = _load_json_lenient(snippet)
        if isinstance(value, dict):
            return value

    # Last-resort lenient parse from widest braces.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        value = _load_json_lenient(text[start : end + 1])
        if isinstance(value, dict):
            return value

    # Truncated-object recovery: salvage valid prefix and auto-close braces.
    repaired = _repair_truncated_json_object(text)
    if repaired:
        value = _load_json_lenient(repaired)
        if isinstance(value, dict):
            LOGGER.warning("LLM JSON recovered from truncated response.")
            return value
    return None


def _extract_balanced_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _load_json_lenient(snippet: str) -> Optional[Any]:
    candidate = snippet.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Remove trailing commas before } or ].
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Repair Python-like JSON (single quotes, None/True/False).
    repaired = candidate
    repaired = re.sub(r"\bNone\b", "null", repaired)
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    try:
        value = ast.literal_eval(candidate)
        return value
    except Exception:
        return None


def _repair_truncated_json_object(text: str) -> Optional[str]:
    """
    Try to recover a usable JSON object from truncated model output.
    Strategy:
    1) Start from first '{'
    2) Iteratively trim unfinished tail at last top-level comma
    3) Auto-close braces/brackets
    """
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    candidate = text[start:].strip()
    candidate = re.sub(r"```(?:json)?", "", candidate, flags=re.I).strip()

    for _ in range(20):
        closed = _auto_close_json(candidate)
        closed = re.sub(r",\s*([}\]])", r"\1", closed)
        if _load_json_lenient(closed) is not None:
            return closed

        cut = _last_top_level_comma(candidate)
        if cut is None:
            break
        candidate = candidate[:cut].rstrip().rstrip(",")
        if not candidate.startswith("{"):
            break

    return None


def _auto_close_json(text: str) -> str:
    stack: List[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()

    out = text
    if in_string:
        out += '"'
    for opener in reversed(stack):
        out += "}" if opener == "{" else "]"
    return out


def _last_top_level_comma(text: str) -> Optional[int]:
    in_string = False
    escaped = False
    depth_obj = 0
    depth_arr = 0
    last: Optional[int] = None

    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth_obj += 1
        elif ch == "}":
            depth_obj = max(0, depth_obj - 1)
        elif ch == "[":
            depth_arr += 1
        elif ch == "]":
            depth_arr = max(0, depth_arr - 1)
        elif ch == "," and depth_obj == 1 and depth_arr == 0:
            last = i

    return last


def _extract_response_text(response: Any) -> str:
    """
    Extract text robustly from Google GenAI response object.
    """
    if hasattr(response, "text") and isinstance(response.text, str) and response.text.strip():
        return response.text

    pieces: List[str] = []
    candidates = getattr(response, "candidates", None)
    if isinstance(candidates, list):
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            if isinstance(parts, list):
                for part in parts:
                    txt = getattr(part, "text", None)
                    if isinstance(txt, str) and txt.strip():
                        pieces.append(txt)
    if pieces:
        return "\n".join(pieces)
    return ""


def _merge_with_fallback(
    llm_data: Dict[str, Any],
    structured_data: Dict[str, Any],
    party_segmentation: Optional[Dict[str, Any]] = None,
    extracted_text: str = "",
) -> Dict[str, Any]:
    out = dict(llm_data)
    fallback = _structured_to_final(structured_data)

    for key, value in fallback.items():
        if key not in out or out[key] in (None, "", [], {}):
            out[key] = value

    out["productsArray"] = _merge_array_objects(
        llm_items=out.get("productsArray"),
        fallback_items=fallback["productsArray"],
        key_order=[
            "productName",
            "Product_HSN_code",
            "Product_Quantity",
            "Product_Unit",
            "Product_Rate",
            "Product_Amount",
        ],
    )
    out["Expenses"] = _merge_array_objects(
        llm_items=out.get("Expenses"),
        fallback_items=fallback["Expenses"],
        key_order=["Expense_Name", "Expense_Percentage", "Expense_Amount"],
    )

    # Hard rule: productsArray is mandatory and should preserve strong extracted rows.
    out["productsArray"] = _enforce_products_mandatory(out["productsArray"], fallback["productsArray"])

    _apply_party_gstin_guardrails(out, structured_data, party_segmentation)
    return normalize_final_invoice_json(
        final_json=out,
        structured_data=structured_data,
        extracted_text=extracted_text,
    )


def _apply_party_gstin_guardrails(
    out: Dict[str, Any],
    structured_data: Dict[str, Any],
    party_segmentation: Optional[Dict[str, Any]],
) -> None:
    supplier_gst = _norm_gst(out.get("Supplier_GST"))
    customer_gst = _norm_gst(out.get("Customer_GSTIN"))

    segmented_supplier = None
    segmented_customer = None
    if isinstance(party_segmentation, dict):
        supplier = party_segmentation.get("supplier", {})
        customer = party_segmentation.get("customer", {})
        if isinstance(supplier, dict):
            segmented_supplier = _norm_gst(supplier.get("gst"))
        if isinstance(customer, dict):
            segmented_customer = _norm_gst(customer.get("gst"))

    parties = structured_data.get("parties", {}) if isinstance(structured_data.get("parties"), dict) else {}
    structured_supplier = _norm_gst(parties.get("Supplier_GST"))
    structured_customer = _norm_gst(parties.get("Customer_GSTIN"))

    # Prefer segmentation ownership when available.
    if segmented_supplier:
        out["Supplier_GST"] = segmented_supplier
        supplier_gst = segmented_supplier
    elif not supplier_gst and structured_supplier:
        out["Supplier_GST"] = structured_supplier
        supplier_gst = structured_supplier

    if segmented_customer is None:
        # Explicitly absent customer GST in segmented evidence -> force null.
        out["Customer_GSTIN"] = None
        customer_gst = None
    elif segmented_customer:
        out["Customer_GSTIN"] = segmented_customer
        customer_gst = segmented_customer
    elif not customer_gst and structured_customer:
        out["Customer_GSTIN"] = structured_customer
        customer_gst = structured_customer

    # Hard rule: never allow duplicated supplier GST in customer GST field.
    if supplier_gst and customer_gst and supplier_gst == customer_gst:
        out["Customer_GSTIN"] = None


def _norm_gst(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text or text in {"NULL", "NONE", "N/A", "NA"}:
        return None
    if not re.fullmatch(r"[0-9A-Z]{15}", text):
        return None
    return text


def _merge_array_objects(
    llm_items: Any,
    fallback_items: List[Dict[str, Any]],
    key_order: List[str],
) -> List[Dict[str, Any]]:
    """
    Preserve detailed line-items from fallback when LLM output is missing/partial.
    """
    if not isinstance(llm_items, list) or len(llm_items) == 0:
        return fallback_items

    llm_dicts = [item for item in llm_items if isinstance(item, dict)]
    if len(llm_dicts) == 0:
        return fallback_items

    # If LLM dropped line items, trust fallback granularity.
    if len(llm_dicts) < len(fallback_items):
        llm_dicts = llm_dicts + [{} for _ in range(len(fallback_items) - len(llm_dicts))]

    merged: List[Dict[str, Any]] = []
    max_len = max(len(llm_dicts), len(fallback_items))
    for idx in range(max_len):
        llm_obj = llm_dicts[idx] if idx < len(llm_dicts) else {}
        fb_obj = fallback_items[idx] if idx < len(fallback_items) else {}
        out: Dict[str, Any] = {}
        for key in key_order:
            llm_val = llm_obj.get(key) if isinstance(llm_obj, dict) else None
            fb_val = fb_obj.get(key) if isinstance(fb_obj, dict) else None
            out[key] = llm_val if llm_val not in (None, "", [], {}) else fb_val

        # Keep any extra fields returned by LLM without losing base keys.
        if isinstance(llm_obj, dict):
            for k, v in llm_obj.items():
                if k not in out:
                    out[k] = v

        if any(v not in (None, "", [], {}) for v in out.values()):
            merged.append(out)

    return merged if merged else fallback_items


def _enforce_products_mandatory(
    products: Any,
    fallback_products: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not isinstance(products, list) or len(products) == 0:
        return fallback_products

    rows = [item for item in products if isinstance(item, dict)]
    if len(rows) == 0:
        return fallback_products

    def row_strength(row: Dict[str, Any]) -> int:
        score = 0
        if row.get("productName") not in (None, "", "Unknown Item"):
            score += 1
        if row.get("Product_Quantity") not in (None, "", 0):
            score += 1
        if row.get("Product_Rate") not in (None, "", 0):
            score += 1
        if row.get("Product_Amount") not in (None, "", 0):
            score += 1
        if row.get("Product_HSN_code") not in (None, "", "null"):
            score += 1
        return score

    llm_strong = sum(1 for row in rows if row_strength(row) >= 2)
    fb_strong = sum(1 for row in fallback_products if row_strength(row) >= 2)

    # If LLM degraded product quality/count compared to preprocessing, keep preprocessing rows.
    if llm_strong < max(1, fb_strong):
        return fallback_products
    return rows


def _structured_to_final(structured_data: Dict[str, Any]) -> Dict[str, Any]:
    header = structured_data.get("header", {}) if isinstance(structured_data.get("header"), dict) else {}
    parties = structured_data.get("parties", {}) if isinstance(structured_data.get("parties"), dict) else {}
    products = structured_data.get("products", []) if isinstance(structured_data.get("products"), list) else []
    expenses = structured_data.get("expenses", []) if isinstance(structured_data.get("expenses"), list) else []
    totals = structured_data.get("totals", {}) if isinstance(structured_data.get("totals"), dict) else {}

    products_array: List[Dict[str, Any]] = []
    for item in products:
        if not isinstance(item, dict):
            continue
        products_array.append(
            {
                "productName": item.get("productName"),
                "Product_HSN_code": item.get("HSN"),
                "Product_GST_Rate": item.get("gst_rate"),
                "Product_Quantity": item.get("quantity"),
                "Product_Unit": item.get("unit"),
                "Product_DisPer": item.get("discount_percent"),
                "Product_Rate": item.get("rate"),
                "Product_Amount": item.get("amount"),
                "Product_Description": item.get("description"),
                "Product_BatchNo": item.get("batch_no"),
                "Product_ExpDate": item.get("exp_date"),
                "Product_MfgDate": item.get("mfg_date"),
                "Product_MRP": item.get("mrp"),
            }
        )
    if not products_array:
        products_array = [
            {
                "productName": "Unknown Item",
                "Product_HSN_code": None,
                "Product_GST_Rate": None,
                "Product_Quantity": None,
                "Product_Unit": None,
                "Product_DisPer": None,
                "Product_Rate": None,
                "Product_Amount": totals.get("Taxable_Amount"),
                "Product_Description": None,
                "Product_BatchNo": None,
                "Product_ExpDate": None,
                "Product_MfgDate": None,
                "Product_MRP": None,
            }
        ]

    expenses_array = []
    for item in expenses:
        if not isinstance(item, dict):
            continue
        expenses_array.append(
            {
                "Expense_Name": item.get("name"),
                "Expense_Percentage": item.get("percentage"),
                "Expense_Amount": item.get("amount"),
            }
        )
    if not expenses_array:
        expenses_array = [{"Expense_Name": "Blank", "Expense_Percentage": 0.0, "Expense_Amount": 0.0}]

    supplier_name = parties.get("SupplierName")
    supplier_first_word = None
    if isinstance(supplier_name, str) and supplier_name.strip():
        supplier_first_word = supplier_name.strip().split()[0]

    return {
        "Customer_Name": parties.get("Customer_Name"),
        "SupplierName": parties.get("SupplierName"),
        "Customer_address": parties.get("Customer_address"),
        "Supplier_address": parties.get("Supplier_Address"),
        "Customer_GSTIN": parties.get("Customer_GSTIN"),
        "Supplier_GST": parties.get("Supplier_GST"),
        "Invoice_Number": header.get("Invoice_Number"),
        "Invoice_Date": header.get("Invoice_Date"),
        "SGST_Amount": totals.get("SGST_Amount"),
        "CGST_Amount": totals.get("CGST_Amount"),
        "IGST_Amount": totals.get("IGST_Amount"),
        "Total_Expenses": totals.get("Total_Expenses"),
        "Taxable_Amount": totals.get("Taxable_Amount"),
        "Total_Amount": totals.get("Total_Amount"),
        "Vehicle_Number": parties.get("Vehicle_Number"),
        "Bank_Name": parties.get("Bank_Name"),
        "bank_account_number": parties.get("bank_account_number"),
        "IFSCCode": parties.get("IFSCCode"),
        "Email": parties.get("Email"),
        "Phone": parties.get("Phone"),
        "Supplier_First_Word": supplier_first_word,
        "productsArray": products_array,
        "Expenses": expenses_array,
        "_meta": {
            "status": "Success",
            "source_file": None,
            "model": None,
            "processed_at": None,
            "duration_ms": None,
            "doc_type": "invoice",
        },
        "tags": [],
    }


TARGET_TOP_LEVEL_KEYS: List[str] = [
    "Customer_Name",
    "SupplierName",
    "Customer_address",
    "Supplier_address",
    "Customer_GSTIN",
    "Supplier_GST",
    "Invoice_Number",
    "Invoice_Date",
    "SGST_Amount",
    "CGST_Amount",
    "IGST_Amount",
    "Total_Expenses",
    "Taxable_Amount",
    "Total_Amount",
    "Vehicle_Number",
    "Bank_Name",
    "bank_account_number",
    "IFSCCode",
    "Email",
    "Phone",
    "Supplier_First_Word",
    "productsArray",
    "Expenses",
    "_meta",
    "tags",
]

PRODUCT_KEYS: List[str] = [
    "productName",
    "Product_HSN_code",
    "Product_GST_Rate",
    "Product_Quantity",
    "Product_Rate",
    "Product_Unit",
    "Product_DisPer",
    "Product_Amount",
    "Product_Description",
    "Product_BatchNo",
    "Product_ExpDate",
    "Product_MfgDate",
    "Product_MRP",
]

EXPENSE_KEYS: List[str] = ["Expense_Name", "Expense_Percentage", "Expense_Amount"]


def normalize_final_invoice_json(
    final_json: Dict[str, Any],
    structured_data: Dict[str, Any],
    extracted_text: str,
) -> Dict[str, Any]:
    fallback = _structured_to_final(structured_data)
    rows = _ocr_rows(structured_data)
    extracted_text = _strip_noise_lines(extracted_text or "")
    out: Dict[str, Any] = {}

    for key in TARGET_TOP_LEVEL_KEYS:
        value = final_json.get(key) if isinstance(final_json, dict) else None
        if value in (None, "", [], {}) and key in fallback:
            value = fallback.get(key)
        out[key] = _normalize_nullish(value)

    # Deterministic invoice number: integer-like token nearest to "invoice no".
    out["Invoice_Number"] = _normalize_invoice_number(
        current_value=out.get("Invoice_Number"),
        extracted_text=extracted_text or "",
        fallback_value=fallback.get("Invoice_Number"),
        structured_data=structured_data,
    )
    if _is_date_like_token(str(out.get("Invoice_Number") or "")):
        out["Invoice_Number"] = None
    if str(out.get("Invoice_Date") or "").strip() == str(out.get("Invoice_Number") or "").strip():
        out["Invoice_Number"] = None

    # Deterministic total: prefer G.Total / Grand Total line, fallback to highest amount.
    g_total_amt = _extract_gtotal_amount(extracted_text or "")
    if g_total_amt is not None:
        out["Total_Amount"] = g_total_amt
    else:
        max_amt = _extract_highest_amount(extracted_text or "")
        if max_amt is not None:
            out["Total_Amount"] = max_amt
    if out.get("Taxable_Amount") in (None, ""):
        out["Taxable_Amount"] = _normalize_nullish(fallback.get("Taxable_Amount"))

    parties = structured_data.get("parties", {}) if isinstance(structured_data.get("parties"), dict) else {}
    seg = structured_data.get("party_segmentation", {}) if isinstance(structured_data.get("party_segmentation"), dict) else {}
    seg_supplier = seg.get("supplier", {}) if isinstance(seg.get("supplier"), dict) else {}
    seg_customer = seg.get("customer", {}) if isinstance(seg.get("customer"), dict) else {}
    supplier_candidates = parties.get("_supplier_candidates", []) if isinstance(parties.get("_supplier_candidates"), list) else []
    customer_candidates = parties.get("_customer_candidates", []) if isinstance(parties.get("_customer_candidates"), list) else []
    supplier_c1 = supplier_candidates[0] if len(supplier_candidates) > 0 else None
    supplier_c2 = supplier_candidates[1] if len(supplier_candidates) > 1 else None
    customer_c1 = customer_candidates[0] if len(customer_candidates) > 0 else None
    customer_c2 = customer_candidates[1] if len(customer_candidates) > 1 else None

    # Guardrail: never keep label-like/address-like values as party names.
    out["SupplierName"] = _pick_first_valid_name(
        supplier_c1,
        supplier_c2,
        out.get("SupplierName"),
        seg_supplier.get("name"),
        parties.get("SupplierName"),
        fallback.get("SupplierName"),
        _extract_supplier_name_from_text(extracted_text or ""),
    )
    out["Customer_Name"] = _pick_first_valid_name(
        customer_c1,
        customer_c2,
        out.get("Customer_Name"),
        seg_customer.get("name"),
        parties.get("Customer_Name"),
        fallback.get("Customer_Name"),
        _extract_customer_name_from_text(extracted_text or ""),
    )
    out["Supplier_First_Word"] = (
        str(out["SupplierName"]).split()[0]
        if out.get("SupplierName")
        else None
    )

    out["Supplier_address"] = _sanitize_address(out.get("Supplier_address"))
    out["Customer_address"] = _sanitize_address(out.get("Customer_address"))
    if not out.get("Supplier_address"):
        out["Supplier_address"] = _sanitize_address(seg_supplier.get("address"))
    if not out.get("Customer_address"):
        out["Customer_address"] = _sanitize_address(seg_customer.get("address"))
    if not out.get("Supplier_address"):
        out["Supplier_address"] = _sanitize_address(
            _extract_supplier_address_from_text(extracted_text or "", out.get("SupplierName"))
        )
    if not out.get("Customer_address"):
        out["Customer_address"] = _sanitize_address(
            _extract_customer_address_from_text(extracted_text or "", out.get("Customer_Name"))
        )

    _apply_party_ownership_lock(out, extracted_text or "")

    supplier_from_rows, supplier_addr_rows = _extract_party_block_from_rows(rows, role="supplier")
    customer_from_rows, customer_addr_rows = _extract_party_block_from_rows(rows, role="customer")
    if supplier_from_rows and not _is_invalid_party_name(supplier_from_rows):
        cur_supplier = out.get("SupplierName")
        cur_supplier_bad = _is_invalid_party_name(cur_supplier)
        if cur_supplier_bad or len(str(supplier_from_rows).strip()) > len(str(cur_supplier or "").strip()):
            out["SupplierName"] = supplier_from_rows
    if customer_from_rows and not _is_invalid_party_name(customer_from_rows):
        cur_customer = out.get("Customer_Name")
        cur_customer_bad = _is_invalid_party_name(cur_customer)
        if cur_customer_bad or len(str(customer_from_rows).strip()) > len(str(cur_customer or "").strip()):
            out["Customer_Name"] = customer_from_rows
    if supplier_addr_rows:
        cur_sa = _sanitize_address(out.get("Supplier_address"))
        new_sa = _sanitize_address(supplier_addr_rows)
        if new_sa and (not cur_sa or len(new_sa) > len(cur_sa)):
            out["Supplier_address"] = new_sa
    if customer_addr_rows:
        cur_ca = _sanitize_address(out.get("Customer_address"))
        new_ca = _sanitize_address(customer_addr_rows)
        if new_ca and (not cur_ca or len(new_ca) > len(cur_ca)):
            out["Customer_address"] = new_ca
    out["Supplier_address"] = _format_address_human(out.get("Supplier_address"))
    out["Customer_address"] = _format_address_human(out.get("Customer_address"))

    # Final GST ownership guardrail (never same; customer GST only with customer-side evidence).
    supplier_gst_evidence = _extract_supplier_gst_from_text(extracted_text or "")
    customer_gst_evidence = _extract_customer_gst_from_text(extracted_text or "")
    supplier_gst = supplier_gst_evidence or _norm_gst(out.get("Supplier_GST"))
    out["Supplier_GST"] = supplier_gst
    if customer_gst_evidence and customer_gst_evidence != supplier_gst:
        out["Customer_GSTIN"] = customer_gst_evidence
    else:
        out["Customer_GSTIN"] = None

    # Standardize products and enforce quantity/rate/amount consistency.
    out["productsArray"] = _normalize_products(out.get("productsArray"), fallback.get("productsArray", []))
    out["Expenses"] = _normalize_expenses(out.get("Expenses"), fallback.get("Expenses", []))
    row_products = _extract_products_from_rows(rows)
    if row_products:
        out["productsArray"] = row_products
    row_expenses = _extract_expenses_from_rows(rows)
    if row_expenses:
        out["Expenses"] = row_expenses
        out["Total_Expenses"] = round(sum(_to_float(item.get("Expense_Amount")) or 0.0 for item in row_expenses), 2)
    _apply_tax_guardrails(out, extracted_text or "")
    _apply_inclusive_tax_product_rule(out, extracted_text or "")
    out["Bank_Name"] = _resolve_bank_name(
        current_value=out.get("Bank_Name"),
        extracted_text=extracted_text or "",
    )
    out["IFSCCode"] = _normalize_ifsc(out.get("IFSCCode"))
    out["Email"] = _normalize_email(out.get("Email"))
    out["_meta"] = _normalize_meta(out.get("_meta"), fallback.get("_meta"))
    out["tags"] = _build_tags(out)

    # Mandatory products field.
    if not out["productsArray"]:
        out["productsArray"] = fallback.get("productsArray", [])
    if not out["productsArray"]:
        out["productsArray"] = [_blank_product()]
    if not out["Expenses"]:
        out["Expenses"] = [dict((k, None) for k in EXPENSE_KEYS)]
        out["Expenses"][0]["Expense_Name"] = "Blank"
        out["Expenses"][0]["Expense_Percentage"] = 0.0
        out["Expenses"][0]["Expense_Amount"] = 0.0

    _coerce_final_output_types(out)

    for key in TARGET_TOP_LEVEL_KEYS:
        if key not in out:
            out[key] = None
    return out


def _normalize_products(value: Any, fallback_products: Any) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    llm_src = value if isinstance(value, list) else []
    fb_src = fallback_products if isinstance(fallback_products, list) else []
    src = llm_src if llm_src else fb_src
    if not isinstance(src, list):
        src = []
    max_len = max(len(src), len(fb_src))
    for i in range(max_len):
        raw = src[i] if i < len(src) else {}
        fb = fb_src[i] if i < len(fb_src) and isinstance(fb_src[i], dict) else {}
        row = dict(raw) if isinstance(raw, dict) else {}
        fixed = {key: _normalize_nullish(row.get(key)) for key in PRODUCT_KEYS}
        name = str(fixed.get("productName") or "").lower()
        if any(k in name for k in PRODUCT_REJECT_TOKENS):
            continue
        if _contains_reject_token(name):
            continue
        _apply_product_fallback_if_unrealistic(fixed, fb)
        _repair_product_numbers(fixed)
        items.append(fixed)
    return items


def _apply_product_fallback_if_unrealistic(row: Dict[str, Any], fb: Dict[str, Any]) -> None:
    if not isinstance(fb, dict) or not fb:
        return
    r = _to_float(row.get("Product_Rate"))
    a = _to_float(row.get("Product_Amount"))
    q = _to_float(row.get("Product_Quantity"))
    h = _to_float(row.get("Product_HSN_code"))
    fr = _to_float(fb.get("Product_Rate"))
    fa = _to_float(fb.get("Product_Amount"))

    if r is not None and h is not None and abs(r - h) < 0.5 and fr not in (None, 0):
        row["Product_Rate"] = fr
    if a is not None and h is not None and abs(a - h) < 0.5 and fa not in (None, 0):
        row["Product_Amount"] = fa

    # Very large rate/amount in item rows are likely OCR contamination for HSN/summary lines.
    if r is not None and fr not in (None, 0) and r > 1_000_000:
        row["Product_Rate"] = fr
    if a is not None and fa not in (None, 0) and a > 1_000_000:
        row["Product_Amount"] = fa

    # If qty=1 and both values present but one is implausibly larger, trust fallback.
    if q == 1 and r is not None and a is not None and r > (a * 20) and fr not in (None, 0):
        row["Product_Rate"] = fr


def _normalize_expenses(value: Any, fallback_expenses: Any) -> List[Dict[str, Any]]:
    src = value if isinstance(value, list) and value else fallback_expenses
    if not isinstance(src, list):
        src = []
    items: List[Dict[str, Any]] = []
    for raw in src:
        row = dict(raw) if isinstance(raw, dict) else {}
        fixed = {key: _normalize_nullish(row.get(key)) for key in EXPENSE_KEYS}
        amount = _to_float(fixed.get("Expense_Amount"))
        name = str(fixed.get("Expense_Name") or "").lower()
        if "amount chargeable" in name:
            continue
        if any(k in name for k in ("cgst", "sgst", "igst", "tax", "% %")):
            continue
        if any(tok in name for tok in EXPENSE_BANNED_TOKENS):
            continue
        if not re.search(r"[a-z]", name):
            continue
        if amount is not None and amount >= 10_000_000 and not any(k in name for k in ("freight", "fee", "labour", "commission", "round")):
            continue
        items.append(fixed)
    return items


def _normalize_meta(value: Any, fallback_meta: Any) -> Dict[str, Any]:
    base = {
        "status": "Success",
        "source_file": None,
        "model": None,
        "processed_at": None,
        "duration_ms": None,
        "doc_type": "invoice",
    }
    src = dict(fallback_meta) if isinstance(fallback_meta, dict) else {}
    if isinstance(value, dict):
        src.update(value)
    for k, v in base.items():
        src.setdefault(k, v)
    src["status"] = "Success"
    return src


def _format_address_human(value: Any) -> Optional[str]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw or raw.lower() in {"null", "none", "na", "n/a"}:
        return None
    parts = [re.sub(r"\s+", " ", p).strip(" ,:-|") for p in raw.split("|")]
    parts = [p for p in parts if p]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} {parts[1]}".strip()
    head = f"{parts[0]} {parts[1]}".strip()
    tail = ", ".join(parts[2:])
    return f"{head}, {tail}".strip(" ,")


def _normalize_nullish(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"", "null", "none", "n/a", "na"}:
            return None
        return stripped
    return value


def _blank_product() -> Dict[str, Any]:
    return {
        "productName": "Unknown Item",
        "Product_HSN_code": None,
        "Product_GST_Rate": None,
        "Product_Quantity": None,
        "Product_Rate": None,
        "Product_Unit": None,
        "Product_DisPer": None,
        "Product_Amount": None,
        "Product_Description": None,
        "Product_BatchNo": None,
        "Product_ExpDate": None,
        "Product_MfgDate": None,
        "Product_MRP": None,
    }


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).replace(",", "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _repair_product_numbers(row: Dict[str, Any]) -> None:
    qty = _to_float(row.get("Product_Quantity"))
    rate = _to_float(row.get("Product_Rate"))
    amt = _to_float(row.get("Product_Amount"))
    hsn_num = _to_float(row.get("Product_HSN_code"))

    if rate is not None and hsn_num is not None and abs(rate - hsn_num) < 0.5 and qty not in (None, 0) and amt is not None:
        rate = round(amt / qty, 2)

    if qty is not None:
        row["Product_Quantity"] = qty
    if rate is not None:
        row["Product_Rate"] = rate
    if amt is not None:
        row["Product_Amount"] = amt

    # Never recompute Product_Amount when extracted amount already exists.
    # Only fill missing values conservatively.
    if qty is not None and amt is not None and (rate is None or rate == 0):
        if qty != 0:
            row["Product_Rate"] = round(amt / qty, 2)
    elif rate is not None and amt is not None and (qty is None or qty == 0):
        if rate != 0:
            row["Product_Quantity"] = round(amt / rate, 3)


def _extract_invoice_no_near_keyword(text: str) -> Optional[str]:
    if not text:
        return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for i, line in enumerate(lines):
        low = line.lower()
        if "invoice no" in low or "inv no" in low or re.search(r"\binvoice\b", low):
            window = " ".join(lines[i : min(len(lines), i + 4)])
            for m in re.finditer(r"\b[A-Z0-9]{1,6}[\/\-][A-Z0-9]{1,6}(?:[\/\-][A-Z0-9]{1,8})+\b", window, re.I):
                token = m.group(0).strip(" .,:;|")
                if _is_valid_invoice_token(token):
                    return token
    for i, line in enumerate(lines):
        low = line.lower()
        if "invoice" not in low:
            continue
        window = " ".join(lines[i : min(len(lines), i + 2)])
        # Prefer token immediately after invoice label.
        m = re.search(r"(?:invoice|inv)\s*(?:no|number|#)?\s*[:\-]?\s*([A-Z0-9][A-Z0-9\/\-]{0,31})\b", window, re.I)
        if m:
            token = m.group(1).strip(" .,:;|")
            if _is_valid_invoice_token(token):
                return token
        # Conservative fallback: only plain integer next to explicit invoice-no labels.
        if re.search(r"(?:invoice|inv)\s*(?:no|number|#)", window, re.I):
            for m_int in re.finditer(r"\b([0-9]{1,12})\b", window):
                token = m_int.group(1)
                prefix = window[max(0, m_int.start() - 16) : m_int.start()].lower()
                if "note" in prefix:
                    continue
                if _is_valid_invoice_token(token):
                    return token
    return None


def _normalize_invoice_number(
    current_value: Any,
    extracted_text: str,
    fallback_value: Any,
    structured_data: Dict[str, Any],
) -> Optional[str]:
    def as_invoice_token(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if _is_valid_invoice_token(text):
            return text
        # Salvage candidate token fragments only if still invoice-like and not date-like.
        for m in re.finditer(r"\b([A-Z0-9][A-Z0-9\/\-]{0,31})\b", text, re.I):
            token = m.group(1).strip()
            if _is_valid_invoice_token(token):
                return token
        return None

    from_rows = _extract_invoice_no_from_rows(structured_data.get("_ocr_rows"))
    if from_rows:
        return from_rows

    from_text = _extract_invoice_no_near_keyword(extracted_text)
    if from_text:
        return from_text

    current_num = as_invoice_token(current_value)
    if current_num:
        return current_num

    fallback_num = as_invoice_token(fallback_value)
    if fallback_num:
        return fallback_num
    return None


def _extract_highest_amount(text: str) -> Optional[float]:
    if not text:
        return None
    nums = re.findall(r"-?\d{1,3}(?:,\d{2,3})*(?:\.\d+)?|-?\d+(?:\.\d+)?", text)
    values: List[float] = []
    for token in nums:
        value = _to_float(token)
        if value is None:
            continue
        digit_count = len(re.sub(r"\D", "", token))
        # Skip long numeric ids/account numbers.
        if digit_count > 9 and "," not in token and "." not in token:
            continue
        # Ignore tiny integers likely row numbers/qty.
        if abs(value) < 100:
            continue
        values.append(value)
    if not values:
        return None
    return max(values)


def _extract_gtotal_amount(text: str) -> Optional[float]:
    if not text:
        return None
    for line in text.splitlines():
        low = line.lower()
        if "g. total" in low or "g total" in low or "grand total" in low:
            val = _largest_amount_in_line(line)
            if val is not None:
                return val
    return None


def _extract_invoice_no_from_rows(rows: Any) -> Optional[str]:
    if not isinstance(rows, list) or not rows:
        return None

    def row_text(row: Dict[str, Any]) -> str:
        return str(row.get("row_text", "")).strip()

    # Primary: locate header row having "Invoice No" and "Dated", then read next row's number in that column band.
    for idx, row in enumerate(rows[:40]):
        if not isinstance(row, dict):
            continue
        rt = row_text(row)
        low = rt.lower()
        if "invoice" not in low or "dated" not in low:
            continue

        invoice_x = None
        dated_x = None
        cells = row.get("cells", [])
        if isinstance(cells, list):
            for cell in cells:
                if not isinstance(cell, dict):
                    continue
                txt = str(cell.get("text", "")).lower()
                x = cell.get("x")
                if isinstance(x, (int, float)):
                    if "invoice" in txt and ("no" in txt or "number" in txt):
                        invoice_x = float(x)
                    if "dated" in txt:
                        dated_x = float(x)

        for j in range(idx + 1, min(len(rows), idx + 4)):
            nxt = rows[j]
            if not isinstance(nxt, dict):
                continue
            nxt_text = row_text(nxt)
            code_match = re.search(r"\b[A-Z0-9]{1,6}[\/\-][A-Z0-9]{1,6}(?:[\/\-][A-Z0-9]{1,8})+\b", nxt_text, re.I)
            if code_match:
                token = code_match.group(0).strip(" .,:;|")
                if _is_valid_invoice_token(token):
                    return token
            has_date = bool(re.search(r"\b\d{1,2}[-/][A-Za-z0-9]{1,3}[-/]\d{2,4}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", nxt_text))
            if not has_date:
                continue

            # Prefer cell-by-cell extraction in invoice column region.
            n_cells = nxt.get("cells", [])
            candidates: List[Tuple[float, str]] = []
            if isinstance(n_cells, list):
                for c in n_cells:
                    if not isinstance(c, dict):
                        continue
                    c_txt = str(c.get("text", ""))
                    c_x = c.get("x")
                    if not isinstance(c_x, (int, float)):
                        continue
                    for m in re.finditer(r"\b\d{1,8}\b", c_txt):
                        n = m.group(0)
                        x = float(c_x)
                        if not _is_valid_invoice_token(n):
                            continue
                        if invoice_x is not None and dated_x is not None:
                            left = min(invoice_x, dated_x) - 140.0
                            right = max(invoice_x, dated_x) - 20.0
                            if left <= x <= right:
                                candidates.append((abs(x - (invoice_x + dated_x) / 2.0), n))
                        else:
                            candidates.append((0.0, n))
            if candidates:
                candidates.sort(key=lambda item: item[0])
                return candidates[0][1]

    # Secondary: explicit invoice pattern with integer only, reject "note".
    for row in rows[:40]:
        if not isinstance(row, dict):
            continue
        rt = row_text(row)
        low = rt.lower()
        if "note" in low and "invoice" not in low:
            continue
        m = re.search(r"(?:invoice|inv)\s*(?:no|number|#)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\/\-]{0,31})\b", rt, re.I)
        if m:
            token = m.group(1).strip(" .,:;|")
            if _is_valid_invoice_token(token):
                return token
    return None


MONTH_TOKEN_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\b",
    re.I,
)


def _is_date_like_token(token: str) -> bool:
    t = token.strip().lower()
    if not t:
        return False
    if re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", t):
        return True
    if re.fullmatch(r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", t):
        return True
    if re.fullmatch(r"\d{1,2}[-/](?:[a-z]{3,9})[-/]\d{2,4}", t):
        return True
    if re.fullmatch(r"(?:[a-z]{3,9})[-/]\d{1,2}[-/]\d{2,4}", t):
        return True
    if MONTH_TOKEN_RE.search(t) and bool(re.search(r"\d", t)):
        return True
    return False


def _is_valid_invoice_token(token: str) -> bool:
    t = token.strip(" .,:;|")
    if not t:
        return False
    if not any(ch.isdigit() for ch in t):
        return False
    if _is_date_like_token(t):
        return False
    low = t.lower()
    if low in {"note", "invoice", "dated", "date"}:
        return False
    if len(t) > 32:
        return False
    return True


def _normalize_compact(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _contains_reject_token(text: str) -> bool:
    low = text.lower()
    return any(tok in low for tok in REJECT_LINE_TOKENS)


ADDRESS_STOPWORDS = (
    "tax invoice",
    "gst invoice",
    "invoice no",
    "buyer (bill to)",
    "consignee (ship to)",
    "buyer's order no",
    "buyer order no",
    "bill to",
    "ship to",
    "consignee",
    "buyer",
    "despatch",
    "dispatch",
    "delivery note",
    "terms of delivery",
    "bill of lading",
    "motor vehicle",
    "vehicle no",
    "supplier's ref",
    "other reference",
    "destination",
    "truck",
    "reference no",
    "other references",
    "state name",
    "code :",
    "gstin",
    "gstin/uin",
)

REJECT_LINE_TOKENS = (
    "gst invoice",
    "tax invoice",
    "declaration",
    "e. & o.e",
    "e&o.e",
    "ack no",
    "ack date",
    "state name",
    "despatch",
    "delivery note",
    "terms of delivery",
    "bill of lading",
    "mode/terms of payment",
    "reference no",
    "authorised signatory",
)

PRODUCT_REJECT_TOKENS = (
    "output igst",
    "output cgst",
    "output sgst",
    "igst @",
    "cgst @",
    "sgst @",
    "tax invoice",
    "declaration",
    "amount chargeable",
    "e. & o.e",
    "state name",
    "ack no",
)

PARTY_LABEL_TOKENS = (
    "consignee",
    "ship to",
    "bill to",
    "buyer",
    "supplier",
    "seller",
    "customer",
    "gst invoice",
    "tax invoice",
    "invoice",
)

ADDRESS_HINT_TOKENS = (
    "road",
    "rd",
    "street",
    "st",
    "lane",
    "ln",
    "nagar",
    "sector",
    "block",
    "plot",
    "near",
    "opp",
    "village",
    "district",
    "state",
    "stand",
    "bus stand",
    "pin",
    "pincode",
    "h.no",
    "h no",
)


def _looks_address_like(value: str) -> bool:
    low = value.lower()
    if any(tok in low for tok in ADDRESS_HINT_TOKENS):
        return True
    if "," in value:
        return True
    if bool(re.search(r"\b\d{2,6}\b", value)):
        return True
    return False


def _is_invalid_party_name(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    low = text.lower()
    if any(tok in low for tok in PARTY_LABEL_TOKENS):
        return True
    if len(text) < 3:
        return True
    if _contains_reject_token(text):
        return True
    if re.search(r"\b(invoice|dated|delivery|reference|terms of)\b", low):
        return True
    if re.fullmatch(r"[a-f0-9]{10,}", text, re.I):
        return True
    if re.search(r"\b[a-f0-9]{10,}\b", text, re.I):
        return True
    if any(ch.isdigit() for ch in text) and len(text) >= 8:
        return True
    if re.search(r"\b[0-9A-Z]{15}\b", text.upper()):
        return True
    if _looks_address_like(text):
        return True
    return False


def _pick_first_valid_name(*values: Any) -> Optional[str]:
    for value in values:
        if not _is_invalid_party_name(value):
            return str(value).strip()
    return None


def _is_company_name_candidate(line: str) -> bool:
    text = (line or "").strip()
    if not text:
        return False
    low = text.lower()
    if any(tok in low for tok in PARTY_LABEL_TOKENS):
        return False
    if re.search(r"\bgstin\b|\binvoice\b|\bdated\b|\bdelivery\b|\border no\b", low):
        return False
    if re.search(r"@[a-z0-9._-]+", low):
        return False
    if re.search(r"\b\d{10}\b", low):
        return False
    if _looks_address_like(text):
        return False
    alpha = sum(1 for c in text if c.isalpha())
    if alpha < 4:
        return False
    return True


def _extract_supplier_name_from_text(text: str) -> Optional[str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return None
    customer_idx = None
    for i, ln in enumerate(lines[:80]):
        low = ln.lower()
        if any(k in low for k in ("buyer", "bill to", "ship to", "consignee", "customer")):
            customer_idx = i
            break
    limit = customer_idx if customer_idx is not None else min(len(lines), 40)
    for ln in lines[:limit]:
        if _is_company_name_candidate(ln):
            return ln
    return None


def _extract_customer_name_from_text(text: str) -> Optional[str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return None
    anchor = None
    for i, ln in enumerate(lines[:120]):
        low = ln.lower()
        if any(k in low for k in ("buyer", "bill to", "ship to", "consignee", "customer")):
            anchor = i
            break
    if anchor is None:
        return None
    for ln in lines[anchor + 1 : min(len(lines), anchor + 8)]:
        if _is_company_name_candidate(ln):
            return ln
    return None


def _sanitize_address(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    text = re.sub(r"\bBuyer\s*\(Bill\s*to\)\b", " ", text, flags=re.I)
    text = re.sub(r"\bConsignee\s*\(Ship\s*to\)\b", " ", text, flags=re.I)
    text = re.sub(r"\bBuyer'?s?\s+Order\s+No\.?\b[^|]*", " ", text, flags=re.I)
    text = re.sub(r"\bBill\s*to\b", " ", text, flags=re.I)
    text = re.sub(r"\bShip\s*to\b", " ", text, flags=re.I)
    text = re.sub(r"\b[A-Z0-9]{3,}(?:\/[A-Z0-9-]+)+\b", " ", text)  # invoice-like tokens (e.g. PAC/01101/26-27)
    text = re.sub(r"\b(?:ph|phone|contact|tel)\s*[:\-]?\s*\+?[0-9,\-\s]{6,}\b", " ", text, flags=re.I)
    text = re.sub(r"\b\d{1,2}[-/][A-Za-z]{3}[-/]\d{2,4}\b", " ", text, flags=re.I)
    text = re.sub(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", " ", text, flags=re.I)
    text = text.replace("\n", " | ")
    chunks = [c.strip(" |") for c in text.split("|") if c.strip(" |")]
    clean_chunks: List[str] = []
    for chunk in chunks:
        low = chunk.lower()
        if _contains_reject_token(chunk):
            continue
        cut_at: Optional[int] = None
        for stop in ADDRESS_STOPWORDS:
            pos = low.find(stop)
            if pos != -1:
                cut_at = pos if cut_at is None else min(cut_at, pos)
        if cut_at is not None:
            chunk = chunk[:cut_at].strip(" ,:-|")
            low = chunk.lower()
            if not chunk:
                continue
        if re.search(r"\b[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{3,4}\b", chunk, re.I):
            continue
        if re.fullmatch(r"\d+", chunk):
            continue
        if len(chunk) < 4:
            continue
        if re.fullmatch(r"[A-Z]{1,3}\d{1,4}", chunk, re.I):
            continue
        if "private limited" in low and not re.search(r"\b(road|rd|street|st|mandi|nagar|sector|block|gate|plot|h\.?no|near)\b", low):
            continue
        clean_chunks.append(chunk)

    if not clean_chunks:
        return None
    return " | ".join(clean_chunks[:3])


def _extract_supplier_address_from_text(text: str, supplier_name: Any) -> Optional[str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return None
    supplier_idx = 0
    best_blocks: List[str] = []
    if isinstance(supplier_name, str) and supplier_name.strip():
        s = supplier_name.strip().lower()
        for i, line in enumerate(lines):
            if s in line.lower():
                supplier_idx = i
                end = min(len(lines), i + 6)
                cand_lines: List[str] = []
                for line2 in lines[i + 1 : end]:
                    cand = _extract_address_candidate(line2)
                    if cand:
                        cand_lines.append(cand)
                if cand_lines:
                    best_blocks.append(" | ".join(cand_lines[:2]))

    if best_blocks:
        best_blocks.sort(key=len, reverse=True)
        return best_blocks[0]

    buyer_idx = None
    for i in range(supplier_idx + 1, min(len(lines), supplier_idx + 20)):
        low = lines[i].lower()
        if any(k in low for k in ("buyer", "bill to", "customer", "consignee")):
            buyer_idx = i
            break
    end = buyer_idx if buyer_idx is not None else min(len(lines), supplier_idx + 10)
    block = lines[supplier_idx + 1 : end]
    if not block:
        return None

    address_like: List[str] = []
    for line in block:
        candidate = _extract_address_candidate(line)
        if candidate:
            address_like.append(candidate)
    if not address_like:
        return None
    return " | ".join(address_like[:2])


def _extract_customer_address_from_text(text: str, customer_name: Any) -> Optional[str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return None
    start = None
    if isinstance(customer_name, str) and customer_name.strip():
        c = customer_name.strip().lower()
        for i, line in enumerate(lines):
            if c in line.lower():
                start = i
                break
    if start is None:
        for i, line in enumerate(lines):
            if "buyer" in line.lower():
                start = i
                break
    if start is None:
        return None
    block = lines[start + 1 : min(len(lines), start + 8)]
    address_like: List[str] = []
    for line in block:
        candidate = _extract_address_candidate(line)
        if candidate:
            address_like.append(candidate)
    if not address_like:
        return None
    return " | ".join(address_like[:2])


def _extract_address_candidate(line: str) -> Optional[str]:
    if not line:
        return None
    work = f" {line} "
    # Remove known non-address snippets but keep remaining text.
    work = re.sub(r"\bGSTIN(?:/UIN)?\s*[:\-]?\s*[0-9A-Z]{15}\b", " ", work, flags=re.I)
    work = re.sub(r"\b[A-Z0-9]{3,}(?:\/[A-Z0-9-]+)+\b", " ", work)  # invoice-like ids
    work = re.sub(r"\b\d{1,2}[-/][A-Za-z]{3}[-/]\d{2,4}\b", " ", work, flags=re.I)
    work = re.sub(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", " ", work, flags=re.I)
    work = re.sub(
        r"\b(Invoice No\.?|Dated|Delivery Note|Mode/Terms of Payment|Despatch Document No\.?|Bill of Lading/LR-RR No\.?|Motor Vehicle No\.?|Terms of Delivery|Despatched through|Destination|State Name|Code)\b",
        " ",
        work,
        flags=re.I,
    )
    work = re.sub(r"\b(Contact|Phone|E-?Mail|Email)\s*[:\-]?\s*[A-Za-z0-9@+._-]+\b", " ", work, flags=re.I)
    work = re.sub(r"\b(?:ph|tel)\s*[:\-]?\s*\+?[0-9,\-\s]{6,}\b", " ", work, flags=re.I)
    work = re.sub(r"\s+", " ", work).strip(" ,:-|")
    work = re.sub(r"\s+\d{1,3}$", "", work).strip(" ,:-|")
    if not work:
        return None

    # Reject mostly technical/logistics residues.
    low = work.lower()
    if any(k in low for k in ("truck", "hr", "vehicle", "authorised", "signatory", "declaration")):
        return None
    if re.fullmatch(r"\d+", work):
        return None
    if len(work) < 6:
        return None

    # Keep likely address spans (contain location tokens or uppercase words with spaces).
    if re.search(r"\b(road|rd|street|st|mandi|nagar|pur|city|dist|near|plot|gate|gali|sector|block|village|tehsil|stand|opp|compound|market|police)\b", low):
        return work
    if re.search(r"\b[A-Z]{2,}\b", work) and " " in work:
        return work
    return None


def _extract_supplier_gst_from_text(text: str) -> Optional[str]:
    return _best_gstin_from_text(text, prefer_customer=False)


def _extract_customer_gst_from_text(text: str) -> Optional[str]:
    return _best_gstin_from_text(text, prefer_customer=True)


def _best_gstin_from_text(text: str, prefer_customer: bool) -> Optional[str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return None

    buyer_idx = None
    for i, line in enumerate(lines):
        low = line.lower()
        if any(k in low for k in ("buyer", "bill to", "ship to", "consignee", "customer")):
            buyer_idx = i
            break

    if prefer_customer and buyer_idx is not None:
        search_lines = lines[buyer_idx : min(len(lines), buyer_idx + 20)]
    elif not prefer_customer and buyer_idx is not None:
        search_lines = lines[:buyer_idx]
    else:
        search_lines = lines[:40]

    raw_tokens: List[str] = []
    for line in search_lines:
        if "gst" not in line.lower() and not re.search(r"\b[0-9A-Z]{15}\b", line.upper()):
            continue
        raw_tokens.extend(re.findall(r"\b[0-9A-Z]{15}\b", line.upper()))

    if not raw_tokens:
        raw_tokens = re.findall(r"\b[0-9A-Z]{15}\b", (text or "").upper())
    if not raw_tokens:
        return None

    # Try OCR confusion normalization and vote by frequency.
    expanded: List[str] = []
    for token in raw_tokens:
        expanded.extend(_gst_variants(token))
    votes: Dict[str, int] = {}
    for token in expanded:
        if re.fullmatch(r"[0-9A-Z]{15}", token):
            votes[token] = votes.get(token, 0) + 1
    if not votes:
        return None
    best = sorted(votes.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[0][0]
    return best


def _gst_variants(token: str) -> List[str]:
    t = token.strip().upper()
    variants = {t}
    # Common OCR fixes for alnum confusion.
    repl_map = {"O": "0", "I": "1", "L": "1", "S": "5", "B": "8"}
    fixed = "".join(repl_map.get(ch, ch) if idx < 2 or idx >= 12 else ch for idx, ch in enumerate(t))
    variants.add(fixed)
    # PAN-body confusion seen in OCR (F/P).
    if len(t) == 15:
        pan = t[2:12]
        if "F" in pan:
            variants.add(t[:2] + pan.replace("F", "P") + t[12:])
    return list(variants)


def _row_text(row: Dict[str, Any]) -> str:
    if not isinstance(row, dict):
        return ""
    return str(row.get("row_text", "") or "").strip()


def _ocr_rows(structured_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = structured_data.get("_ocr_rows")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _extract_party_block_from_rows(rows: List[Dict[str, Any]], role: str) -> Tuple[Optional[str], Optional[str]]:
    if not rows:
        return None, None
    lines = [_row_text(row) for row in rows]
    lows = [line.lower() for line in lines]

    buyer_idx = None
    for i, low in enumerate(lows[:120]):
        if any(k in low for k in ("buyer (bill to)", "buyer", "bill to", "ship to", "consignee", "customer", "m/s")):
            buyer_idx = i
            break

    def state_from_line(line: str) -> Optional[str]:
        m = re.search(r"state\s*name\s*[|:\s]*([A-Za-z ]+)", line, re.I)
        if not m:
            return None
        value = m.group(1)
        value = re.split(r",\s*code|\bcode\b", value, flags=re.I)[0]
        value = re.sub(r"\s+", " ", value).strip(" :-|,")
        return value if value else None

    def is_good_name(line: str) -> bool:
        cleaned = re.sub(r"\(.*?from.*?\)", " ", line, flags=re.I)
        cleaned = re.sub(r"\bfrom\b", " ", cleaned, flags=re.I)
        cleaned = re.sub(r"\b\d{1,2}[-/][A-Za-z]{3}[-/]\d{2,4}\b", " ", cleaned, flags=re.I)
        cleaned = re.sub(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b", " ", cleaned, flags=re.I)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,:-|")
        low = cleaned.lower()
        if _contains_reject_token(cleaned):
            return False
        if any(k in low for k in ("original", "duplicate", "copy", "recipient")):
            return False
        if any(k in low for k in ("irn", "ack", "invoice", "dated", "gstin", "state name", "delivery", "dispatch", "reference", "contact", "email", "tel:")):
            return False
        if re.search(r"[0-9]{6,}", cleaned):
            return False
        return _is_company_name_candidate(cleaned)

    if role == "customer":
        if buyer_idx is None:
            return None, None
        scan = lines[buyer_idx + 1 : min(len(lines), buyer_idx + 14)]
        name = None
        for line in scan:
            if is_good_name(line):
                name = line.strip()
                break
        addr_parts: List[str] = []
        name_seen = name is None
        for line in scan:
            low = line.lower()
            if any(stop in low for stop in ("contact", "entact", "person :", "description of goods", "hsn/sac", "si description")):
                break
            if any(skip in low for skip in ("dispatch", "delivery note", "dispatched through", "destination")):
                continue
            if "gstin" in low:
                continue
            if not name_seen:
                if name and line.strip() == name.strip():
                    name_seen = True
                continue
            # Skip single-token person names like "MANISH".
            if re.fullmatch(r"[A-Za-z]{3,}", line.strip()):
                continue
            if any(k in low for k in ("roadways", "rahul", "parveen")):
                continue
            st = state_from_line(line)
            if st:
                addr_parts.append(st)
                continue
            cand = _extract_address_candidate(line)
            if cand:
                addr_parts.append(cand)
        dedup_addr: List[str] = []
        for part in addr_parts:
            if part and part not in dedup_addr:
                dedup_addr.append(part)
        address = " | ".join(dedup_addr[:3]) if dedup_addr else None
        return (name, address)

    # Supplier block: top section before buyer label only and near supplier GST.
    end = buyer_idx if buyer_idx is not None else min(len(lines), 40)
    end = min(end, 30)
    scan = lines[:end]
    name = None
    name_idx = None
    # Prefer business-name line nearest to supplier GST anchor.
    gst_anchor = None
    for idx, line in enumerate(scan):
        if "gstin" in line.lower():
            gst_anchor = idx
            break
    if gst_anchor is not None:
        start = max(0, gst_anchor - 8)
        for idx in range(gst_anchor - 1, start - 1, -1):
            line = scan[idx]
            if is_good_name(line):
                name = re.sub(r"\(.*?from.*?\)", " ", line, flags=re.I).strip(" ,:-|")
                name_idx = idx
                break
    if name_idx is None:
        for idx, line in enumerate(scan):
            if is_good_name(line):
                name = re.sub(r"\(.*?from.*?\)", " ", line, flags=re.I).strip(" ,:-|")
                name_idx = idx
                break
    # If a short repeated suffix was selected, prefer a nearby longer business name containing it.
    if name_idx is not None and name:
        for j in range(max(0, name_idx - 6), name_idx):
            prev = re.sub(r"\(.*?from.*?\)", " ", scan[j], flags=re.I).strip(" ,:-|")
            if not prev or not is_good_name(prev):
                continue
            if name.lower() in prev.lower() and len(prev) > len(name):
                name = prev
                name_idx = j
                break
    if name_idx is None:
        return None, None

    addr_parts: List[str] = []
    for line in scan[name_idx + 1 : min(len(scan), name_idx + 12)]:
        low = line.lower()
        if any(stop in low for stop in ("buyer", "bill to", "ship to", "consignee", "description of goods", "hsn/sac")):
            break
        # Skip repeated company-name lines while collecting address.
        if is_good_name(line):
            continue
        if any(skip in low for skip in ("delivery note", "dispatch", "reference", "invoice no", "dated")):
            cand = _extract_address_candidate(line)
            if cand:
                addr_parts.append(cand)
            continue
        if any(skip in low for skip in ("gstin", "contact", "email", "tel:")):
            continue
        st = state_from_line(line)
        if st:
            addr_parts.append(st)
            continue
        cand = _extract_address_candidate(line)
        if cand:
            addr_parts.append(cand)

    dedup_addr: List[str] = []
    for part in addr_parts:
        if part and part not in dedup_addr:
            dedup_addr.append(part)
    address = " | ".join(dedup_addr[:4]) if dedup_addr else None
    return name, address


def _extract_products_from_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    products: List[Dict[str, Any]] = []
    in_table = False
    for row in rows:
        line = _row_text(row)
        low = line.lower()
        if not line:
            continue
        if any(tok in low for tok in ("description", "description of goods")) and any(tok in low for tok in ("qty", "quantity")) and "rate" in low:
            in_table = True
            continue
        if not in_table:
            continue
        if any(
            low.startswith(tok)
            for tok in ("sub total", "subtotal", "cgst", "sgst", "igst", "g. total", "grand total", "total")
        ):
            break

        m_serial = re.match(r"^\s*(\d+)\s+(.*)$", line)
        if not m_serial:
            # Multiline continuation: append to last product name.
            if products and not any(k in low for k in ("gst", "hsn", "qty", "rate", "amount")):
                cont = re.sub(r"^\s*\d+\s*", "", line).strip(" |")
                # Stop merging when line is numeric-only/summary amount.
                if re.fullmatch(r"\d{1,3}(?:,\d{2,3})*(?:\.\d+)?", line.strip()):
                    continue
                if cont and len(cont) > 1:
                    prev_name = str(products[-1]["productName"])
                    m_dec = re.match(r"^\s*(\d+)\s+(.*)$", line.strip())
                    if prev_name.endswith(".") and m_dec:
                        # Merge broken token: "20X2." + "5 ASS LONG" -> "20X2.5 ASS LONG"
                        products[-1]["productName"] = f"{prev_name}{m_dec.group(1)} {m_dec.group(2)}".strip()
                    else:
                        products[-1]["productName"] = f"{prev_name} {cont}".strip()
            continue

        body = m_serial.group(2)
        hsn_match = re.search(r"\b(\d{6,8})\b", body)
        if not hsn_match:
            if products and len(body.strip()) > 1:
                cont = body.strip(" |")
                if cont and not any(k in cont.lower() for k in ("sgst", "cgst", "igst", "total")):
                    prev_name = str(products[-1]["productName"])
                    lead_num = m_serial.group(1)
                    if prev_name.endswith(".") and lead_num:
                        products[-1]["productName"] = f"{prev_name}{lead_num} {cont}".strip()
                    else:
                        products[-1]["productName"] = f"{prev_name} {cont}".strip()
            continue

        desc = body[: hsn_match.start()].strip(" |")
        parts = [part.strip() for part in body.split("|") if part.strip()]
        qty = None
        unit = None
        rate = None
        amount = None

        qty_match = re.search(r"\b(\d+(?:\.\d+)?)\s*(pcs|nos|kg|qt|qtl|mt|ltr|unit)\b\.?", body, re.I)
        if qty_match:
            qty = _to_float(qty_match.group(1))
            unit = qty_match.group(2)
        percents = [_to_float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", body)]
        percents = [p for p in percents if p is not None]
        gst_rate = percents[0] if percents else 0.0
        dis_per = 0.0
        if len(percents) >= 2:
            candidates = [p for p in percents[1:] if p is not None and 0.0 < p <= 25.0]
            dis_per = (candidates[-1] if candidates else 0.0)

        if len(parts) >= 2:
            rate_candidates = re.findall(r"\d{1,3}(?:,\d{2,3})*(?:\.\d+)?|\d+(?:\.\d+)?", parts[-2])
            if rate_candidates:
                rate = _to_float(rate_candidates[0])
        amt_candidates = re.findall(r"\d{1,3}(?:,\d{2,3})*(?:\.\d+)?|\d+(?:\.\d+)?", parts[-1] if parts else "")
        if amt_candidates:
            amount = _to_float(amt_candidates[-1])

        if not desc or amount is None:
            continue
        products.append(
            {
                "productName": re.sub(r"\s+", " ", desc).strip(),
                "Product_HSN_code": hsn_match.group(1),
                "Product_GST_Rate": gst_rate if gst_rate is not None else 0.0,
                "Product_Quantity": qty if qty is not None else 0.0,
                "Product_Rate": rate if rate is not None else 0.0,
                "Product_Unit": unit or "Nos",
                "Product_DisPer": dis_per,
                "Product_Amount": amount,
                "Product_Description": "null",
                "Product_BatchNo": "null",
                "Product_ExpDate": "null",
                "Product_MfgDate": "null",
                "Product_MRP": 0.0,
            }
        )
    return products


def _extract_expenses_from_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    expenses: List[Dict[str, Any]] = []
    seen_tax_line = False
    for row in rows:
        line = _row_text(row)
        low = line.lower()
        if not line:
            continue
        if low.startswith("sgst") or low.startswith("cgst") or low.startswith("igst"):
            seen_tax_line = True
            continue
        if not seen_tax_line:
            continue
        if low.startswith("total"):
            break
        nums = re.findall(r"\d{1,3}(?:,\d{2,3})*(?:\.\d+)?|\d+(?:\.\d+)?", line)
        if not nums:
            continue
        amount = _to_float(nums[-1])
        if amount is None:
            continue
        pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", line)
        pct = _to_float(pct_match.group(1)) if pct_match else 0.0
        name = re.sub(r"\|", " ", line)
        name = re.sub(r"\d{1,3}(?:,\d{2,3})*(?:\.\d+)?|\d+(?:\.\d+)?", " ", name)
        name = re.sub(r"\s+", " ", name).strip(" :-")
        if not name:
            continue
        low_name = name.lower()
        if any(k in low_name for k in ("cgst", "sgst", "igst", "tax", "total")):
            continue
        if any(tok in low_name for tok in EXPENSE_BANNED_TOKENS):
            continue
        if not re.search(r"[a-z]", low_name):
            continue
        expenses.append(
            {
                "Expense_Name": name,
                "Expense_Percentage": pct if pct is not None else 0.0,
                "Expense_Amount": amount,
            }
        )
    return expenses

def _extract_tax_amounts_from_text(text: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    if not text:
        return values
    for line in text.splitlines():
        low = line.lower()
        compact = re.sub(r"\s+", "", low)
        if "sgst" in compact:
            amt = _largest_amount_in_line(line)
            if amt is not None:
                values["SGST_Amount"] = amt
        if "cgst" in compact:
            amt = _largest_amount_in_line(line)
            if amt is not None:
                values["CGST_Amount"] = amt
        if "igst" in compact:
            amt = _largest_amount_in_line(line)
            if amt is not None:
                values["IGST_Amount"] = amt
    return values


def _largest_amount_in_line(line: str) -> Optional[float]:
    nums = re.findall(r"\d{1,3}(?:,\d{2,3})*(?:\.\d+)?|\d+(?:\.\d+)?", line)
    values: List[float] = []
    for token in nums:
        val = _to_float(token)
        if val is None:
            continue
        if val < 1:
            continue
        digit_count = len(re.sub(r"\D", "", token))
        if digit_count > 9 and "," not in token and "." not in token:
            # likely account/HSN/id-like token
            continue
        values.append(val)
    if not values:
        return None
    return max(values)


def _collect_hsn_candidates(products: Any) -> set[str]:
    out: set[str] = set()
    if not isinstance(products, list):
        return out
    for row in products:
        if not isinstance(row, dict):
            continue
        hsn = row.get("Product_HSN_code")
        if hsn is None:
            continue
        digits = re.sub(r"\D", "", str(hsn))
        if 4 <= len(digits) <= 10:
            out.add(digits)
    return out


def _apply_tax_guardrails(out: Dict[str, Any], extracted_text: str) -> None:
    tax_from_labels = _extract_tax_amounts_from_text(extracted_text)
    hsn_set = _collect_hsn_candidates(out.get("productsArray"))
    for field in ("SGST_Amount", "CGST_Amount", "IGST_Amount"):
        if field in tax_from_labels:
            out[field] = tax_from_labels[field]
            continue
        current = out.get(field)
        if current is None:
            out[field] = 0.0
            continue
        val = _to_float(current)
        if val is None:
            out[field] = 0.0
            continue
        digits = re.sub(r"\D", "", str(current))
        if digits in hsn_set:
            # Hard rule: HSN must never populate tax amounts.
            out[field] = 0.0
            continue
        # No clear tax-line evidence -> keep conservative default.
        out[field] = 0.0

    sgst = _to_float(out.get("SGST_Amount")) or 0.0
    cgst = _to_float(out.get("CGST_Amount")) or 0.0
    igst = _to_float(out.get("IGST_Amount")) or 0.0
    # Intra-state sanity: SGST and CGST should match when IGST is not applied.
    if igst == 0.0:
        if sgst > 0 and cgst == 0:
            out["CGST_Amount"] = sgst
        elif cgst > 0 and sgst == 0:
            out["SGST_Amount"] = cgst
        elif sgst > 0 and cgst > 0 and abs(sgst - cgst) > 0.01:
            fixed = max(sgst, cgst)
            out["SGST_Amount"] = fixed
            out["CGST_Amount"] = fixed


def _extract_tax_rate_pair(text: str) -> Tuple[float, float]:
    sgst_rate = 0.0
    cgst_rate = 0.0
    for line in (text or "").splitlines():
        low = line.lower()
        compact = re.sub(r"\s+", "", low)
        if "sgst" in compact:
            m = re.search(r"(\d+(?:\.\d+)?)\s*%", line)
            if m:
                sgst_rate = _to_float(m.group(1)) or sgst_rate
        if "cgst" in compact:
            m = re.search(r"(\d+(?:\.\d+)?)\s*%", line)
            if m:
                cgst_rate = _to_float(m.group(1)) or cgst_rate
    return sgst_rate, cgst_rate


def _apply_inclusive_tax_product_rule(out: Dict[str, Any], extracted_text: str) -> None:
    products = out.get("productsArray")
    if not isinstance(products, list):
        return
    incl_tax = "incl. of tax" in extracted_text.lower() or "incl of tax" in extracted_text.lower()
    sgst_rate, cgst_rate = _extract_tax_rate_pair(extracted_text)
    derived_gst = round((sgst_rate + cgst_rate), 2) if (sgst_rate > 0 and cgst_rate > 0) else 0.0
    if derived_gst == 0.0:
        sgst_amt = _to_float(out.get("SGST_Amount")) or 0.0
        cgst_amt = _to_float(out.get("CGST_Amount")) or 0.0
        taxable = _to_float(out.get("Taxable_Amount")) or 0.0
        if taxable <= 0:
            taxable = sum((_to_float(item.get("Product_Amount")) or 0.0) for item in products if isinstance(item, dict))
        if taxable > 0 and (sgst_amt + cgst_amt) > 0:
            derived_gst = round(((sgst_amt + cgst_amt) / taxable) * 100.0, 2)

    for row in products:
        if not isinstance(row, dict):
            continue
        qty = _to_float(row.get("Product_Quantity"))
        amt = _to_float(row.get("Product_Amount"))
        rate = _to_float(row.get("Product_Rate"))
        gst = _to_float(row.get("Product_GST_Rate")) or 0.0
        taxable_total = _to_float(out.get("Taxable_Amount")) or 0.0

        if incl_tax and qty not in (None, 0) and amt not in (None, 0):
            row["Product_Rate"] = round(amt / qty, 2)
        elif rate in (None, 0) and qty not in (None, 0) and amt not in (None, 0):
            row["Product_Rate"] = round(amt / qty, 2)

        # If OCR placed taxable unit-rate into amount field (common in "Incl. of Tax" layouts), swap.
        if incl_tax and qty not in (None, 0) and amt not in (None, 0) and taxable_total > 0:
            if abs((amt * qty) - taxable_total) <= max(5.0, 0.03 * taxable_total):
                row["Product_Rate"] = round(amt, 2)
                row["Product_Amount"] = round(taxable_total, 2)

        if derived_gst > 0 and (gst == 0.0 or gst < derived_gst):
            row["Product_GST_Rate"] = derived_gst


def _resolve_bank_name(current_value: Any, extracted_text: str) -> Optional[str]:
    lines = [ln.strip() for ln in extracted_text.splitlines() if ln.strip()]
    bank_lines = [ln for ln in lines if re.search(r"\bbank\s*name\s*:|\bbank\s*:", ln, re.I)]

    def pick_from_lines(src_lines: List[str]) -> Optional[str]:
        for line in src_lines:
            if _contains_reject_token(line):
                continue
            m = re.search(r"\bbank(?:\s*name)?\s*:\s*([A-Za-z][A-Za-z&.\s]{1,40}?)(?=\s+A\/?C\b|\s+ACCOUNT\b|\s+IFSC\b|$)", line, re.I)
            if not m:
                continue
            candidate = re.sub(r"\s+", " ", m.group(1)).strip(" :-|")
            candidate = re.split(r"\b(a/c|account|ifsc|branch)\b", candidate, flags=re.I)[0].strip(" :-|")
            if candidate:
                return candidate
        return None

    from_text = pick_from_lines(bank_lines)
    if from_text:
        return from_text

    current = str(current_value).strip() if current_value is not None else ""
    if not current or current.lower() in {"null", "none", "na", "n/a"}:
        return None
    if "bank" not in current.lower():
        return None
    return current


def _apply_party_ownership_lock(out: Dict[str, Any], extracted_text: str) -> None:
    supplier_name = out.get("SupplierName")
    customer_name = out.get("Customer_Name")
    supplier_addr = out.get("Supplier_address")
    customer_addr = out.get("Customer_address")

    if _is_invalid_party_name(supplier_name):
        out["SupplierName"] = None
    if _is_invalid_party_name(customer_name):
        out["Customer_Name"] = None

    s_addr_norm = _normalize_compact(supplier_addr)
    c_addr_norm = _normalize_compact(customer_addr)
    if s_addr_norm and c_addr_norm and s_addr_norm == c_addr_norm:
        # Avoid cross-copy contamination.
        out["Customer_address"] = None

    if out.get("Customer_address"):
        ca = str(out["Customer_address"]).lower()
        # If customer anchor is absent in OCR text, keep conservative null.
        if not any(k in extracted_text.lower() for k in ("buyer", "bill to", "ship to", "consignee", "customer")):
            out["Customer_address"] = None
        elif any(k in ca for k in ("supplier", "seller")):
            out["Customer_address"] = None


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


def _coerce_final_output_types(out: Dict[str, Any]) -> None:
    for key in TOP_STRING_FIELDS:
        out[key] = _to_string_or_null_literal(out.get(key))
    for key in TOP_NUMERIC_FIELDS:
        out[key] = _to_float_or_zero(out.get(key))

    products = out.get("productsArray")
    if not isinstance(products, list):
        products = []
    coerced_products: List[Dict[str, Any]] = []
    for item in products:
        row = dict(item) if isinstance(item, dict) else {}
        for key in PRODUCT_STRING_FIELDS:
            row[key] = _to_string_or_null_literal(row.get(key))
        for key in PRODUCT_NUMERIC_FIELDS:
            row[key] = _to_float_or_zero(row.get(key))
        coerced_products.append(row)
    out["productsArray"] = coerced_products

    expenses = out.get("Expenses")
    if not isinstance(expenses, list):
        expenses = []
    coerced_expenses: List[Dict[str, Any]] = []
    for item in expenses:
        row = dict(item) if isinstance(item, dict) else {}
        for key in EXPENSE_STRING_FIELDS:
            row[key] = _to_string_or_null_literal(row.get(key))
        for key in EXPENSE_NUMERIC_FIELDS:
            row[key] = _to_float_or_zero(row.get(key))
        coerced_expenses.append(row)
    out["Expenses"] = coerced_expenses

    meta = out.get("_meta")
    if not isinstance(meta, dict):
        meta = {}
    out["_meta"] = {
        "status": _to_string_or_null_literal(meta.get("status")),
        "source_file": _to_string_or_null_literal(meta.get("source_file")),
        "model": _to_string_or_null_literal(meta.get("model")),
        "processed_at": _to_string_or_null_literal(meta.get("processed_at")),
        "duration_ms": _to_string_or_null_literal(meta.get("duration_ms")),
        "doc_type": _to_string_or_null_literal(meta.get("doc_type")),
    }


def _to_string_or_null_literal(value: Any) -> str:
    if value is None:
        return "null"
    text = str(value).strip()
    if text == "" or text.lower() in {"null", "none", "na", "n/a"}:
        return "null"
    return text


def _to_float_or_zero(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if text == "" or text.lower() in {"null", "none", "na", "n/a"}:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _normalize_ifsc(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text or text.lower() in {"null", "none", "na", "n/a"}:
        return None
    text = re.sub(r"[^A-Z0-9]", "", text)
    text = text.replace("OOO", "000").replace("OO", "00")
    if len(text) == 11 and re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}", text):
        return text
    return None


def _normalize_email(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "na", "n/a"}:
        return None
    if not re.fullmatch(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
        return None
    return text


def _build_tags(out: Dict[str, Any]) -> List[str]:
    party = _to_string_or_null_literal(out.get("SupplierName"))
    gstin = _to_string_or_null_literal(out.get("Supplier_GST"))
    hs_codes = 0
    products = out.get("productsArray")
    if isinstance(products, list):
        seen = set()
        for row in products:
            if not isinstance(row, dict):
                continue
            code = _to_string_or_null_literal(row.get("Product_HSN_code"))
            if code != "null":
                seen.add(code)
        hs_codes = len(seen)
    return [
        "Doc:invoice",
        f"Party:{party}",
        f"GSTIN:{gstin}",
        "FY:Unknown",
        f"HS:{hs_codes}",
    ]
