"""
LLM Refinement Module - Simple Google GenAI Integration

Uses google-genai SDK directly for basic prompting
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

from google import genai


def load_api_key() -> str:
    """Load API key from environment variables or .env file."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        return api_key.strip().strip('"').strip("'")

    env_path = Path(".env")
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() in {"GOOGLE_API_KEY", "GEMINI_API_KEY"}:
                return value.strip().strip('"').strip("'")

    return ""


def load_invoice_prompt(invoice_type: str) -> str:
    """
    Load the appropriate invoice extraction prompt.

    Args:
        invoice_type: "Purchase Invoice" or "Sales Invoice"

    Returns:
        Prompt text content
    """
    prompt_path = Path("PDF TO TALLY") / "prompts"

    if invoice_type == "Purchase Invoice":
        prompt_file = prompt_path / "Purchase_Invoice.txt"
    else:
        prompt_file = prompt_path / "Sales_Invoice.txt"

    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8")

    raise FileNotFoundError(f"Prompt file not found: {prompt_file}")


def refine_invoice_json_with_llm(
    extracted_text: str,
    extracted_json: Dict[str, Any],
    invoice_type: str,
    model_name: str = "gemini-2.5-flash",
    timeout_seconds: int = 60
) -> Optional[Dict[str, Any]]:
    """
    Refine extracted invoice data using Gemini LLM.

    Args:
        extracted_text: Raw OCR text from document
        extracted_json: Structured data from OCR extraction
        invoice_type: "Purchase Invoice" or "Sales Invoice"
        model_name: Gemini model to use (without "models/" prefix)
        timeout_seconds: Request timeout

    Returns:
        Refined JSON dict if successful, None otherwise
    """

    # Load the invoice-specific prompt
    try:
        prompt_template = load_invoice_prompt(invoice_type)
    except FileNotFoundError as e:
        raise ValueError(f"Invoice prompt loading failed: {str(e)}")

    # Build the user message
    user_message = f"""{prompt_template}

# Extracted OCR Data:
```json
{json.dumps(extracted_json, ensure_ascii=False, indent=2)}
```

# Extracted Text:
```
{extracted_text[:2000]}
```

Please analyze the above invoice data and extract/refine all fields according to the JSON structure specified above. Return ONLY the JSON object, nothing else."""

    # Get API key
    api_key = load_api_key()
    if not api_key:
        raise ValueError("Missing Gemini API key. Set GOOGLE_API_KEY or GEMINI_API_KEY in environment or .env.")

    # Create client
    client = genai.Client(api_key=api_key)

    # Remove "models/" prefix if present
    if model_name.startswith("models/"):
        model_name = model_name[7:]

    # Generate content
    response = client.models.generate_content(
        model=model_name,
        contents=user_message,
        config={
            "temperature": 0.1,
            "top_p": 1,
            "top_k": 32,
            "max_output_tokens": 4096,
        },
    )

    # Extract text from response
    response_text = response.text if hasattr(response, "text") and response.text is not None else ""
    return _parse_json_from_response(response_text)


def _parse_json_from_response(response_text: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Parse JSON from LLM response text.
    Handles markdown code blocks and raw JSON.

    Args:
        response_text: Raw response from LLM (can be None or empty string)

    Returns:
        Parsed JSON dict or None
    """
    if not response_text or not response_text.strip():
        return None

    text = response_text.strip()

    # Try direct JSON parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code blocks
    if "```json" in text:
        try:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                json_str = text[start:end].strip()
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass

    # Try extracting from generic code blocks
    if "```" in text:
        try:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                json_str = text[start:end].strip()
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass

    # Try finding JSON object pattern
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = text[start:end + 1]
            return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        pass

    return None


def validate_invoice_json(
    data: Dict[str, Any],
    invoice_type: str
) -> bool:
    """
    Validate that refined JSON has required fields for invoice type.

    Args:
        data: JSON data to validate
        invoice_type: "Purchase Invoice" or "Sales Invoice"

    Returns:
        True if all required fields present
    """
    required_fields = {
        "Customer_Name",
        "SupplierName",
        "Invoice_Number",
        "Invoice_Date",
        "Total_Amount",
        "productsArray",
        "Expenses",
    }

    return all(field in data for field in required_fields)


if __name__ == "__main__":
    # Example usage
    print("LLM Refinement Module loaded successfully")

    # Check API key
    api_key = load_api_key()
    if api_key:
        print(f"API Key found: {api_key[:20]}...")
    else:
        print("No API key found")
