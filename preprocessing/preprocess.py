from typing import Any, Dict, Optional

from invoice_preprocessor import preprocess_invoice_data


def preprocess_invoice(extracted_json: Optional[Dict[str, Any]], extracted_text: str) -> Dict[str, Any]:
    """
    Convert noisy OCR JSON/text into deterministic structured intermediate JSON.
    """
    return preprocess_invoice_data(extracted_json=extracted_json, extracted_text=extracted_text)

