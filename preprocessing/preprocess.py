from typing import Any, Dict, Optional

from invoice_preprocessor import preprocess_invoice_data
from preprocessing.structure_engine import build_structured_invoice


def preprocess_invoice(extracted_json: Optional[Dict[str, Any]], extracted_text: str) -> Dict[str, Any]:
    """
    Convert noisy OCR JSON/text into deterministic structured intermediate JSON.
    """
    structured = build_structured_invoice(extracted_json=extracted_json, extracted_text=extracted_text)
    # Safety fallback for edge invoices where structure engine yields too little data.
    has_products = isinstance(structured.get("products"), list) and len(structured.get("products", [])) > 0
    has_parties = isinstance(structured.get("parties"), dict) and any(
        structured["parties"].get(k) not in (None, "", "null")
        for k in ("SupplierName", "Customer_Name", "Supplier_GST", "Customer_GSTIN")
    )
    if has_products and has_parties:
        return structured
    return preprocess_invoice_data(extracted_json=extracted_json, extracted_text=extracted_text)
