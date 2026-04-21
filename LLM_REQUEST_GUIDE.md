# LLM Request Module

## Overview

`llm_request.py` is a modular, SDK-based interface for refining invoice extraction with Google Gemini LLM.

**Key Advantage:** Uses official SDKs (google-genai or google-generativeai) instead of REST API calls, avoiding authentication and billing issues.

## Features

✅ **Dual SDK Support**
- Automatic detection of available Gemini SDK
- Falls back gracefully if one SDK is missing
- No manual configuration needed

✅ **Invoice-Specific Prompts**
- Loads Purchase_Invoice.txt or Sales_Invoice.txt automatically
- Uses exact prompt format you defined
- Returns JSON matching your sample formats

✅ **Robust JSON Parsing**
- Handles markdown code blocks
- Extracts JSON from various response formats
- Validates required fields

✅ **Error Handling**
- Clear error messages for missing API keys
- Graceful fallback for prompt loading
- Detailed validation reporting

## Installation

No additional packages needed - uses existing dependencies:
```bash
pip install google-genai google-generativeai
```

## API Reference

### Functions

#### `load_api_key() -> str`
Load Gemini API key from environment or .env file.

**Returns:** API key string or empty string

**Example:**
```python
from llm_request import load_api_key

api_key = load_api_key()
if api_key:
    print("API key loaded successfully")
```

#### `using_new_sdk() -> bool`
Check if google-genai SDK is available.

#### `using_legacy_sdk() -> bool`
Check if google-generativeai SDK is available.

#### `load_invoice_prompt(invoice_type: str) -> str`
Load the appropriate invoice extraction prompt.

**Parameters:**
- `invoice_type`: "Purchase Invoice" or "Sales Invoice"

**Returns:** Prompt text content

**Raises:** `FileNotFoundError` if prompt file not found

**Example:**
```python
from llm_request import load_invoice_prompt

prompt = load_invoice_prompt("Purchase Invoice")
print(f"Prompt length: {len(prompt)} chars")
```

#### `refine_invoice_json_with_llm(...) -> Optional[Dict[str, Any]]`
Refine extracted invoice data using Gemini LLM.

**Parameters:**
- `extracted_text` (str): Raw OCR text from document
- `extracted_json` (Dict): Structured data from OCR extraction
- `invoice_type` (str): "Purchase Invoice" or "Sales Invoice"
- `model_name` (str): Gemini model name (default: "gemini-2.5-flash")
- `timeout_seconds` (int): Request timeout (default: 60)

**Returns:** Refined JSON dict or None if failed

**Raises:** `ValueError` for API/prompt errors

**Example:**
```python
from llm_request import refine_invoice_json_with_llm

refined = refine_invoice_json_with_llm(
    extracted_text="Invoice details...",
    extracted_json={"Customer_Name": "ABC Inc", ...},
    invoice_type="Purchase Invoice",
    model_name="gemini-2.5-flash",
    timeout_seconds=60
)

if refined:
    print(json.dumps(refined, indent=2))
```

#### `validate_invoice_json(data: Dict, invoice_type: str) -> bool`
Validate that JSON has all required invoice fields.

**Parameters:**
- `data`: JSON data to validate
- `invoice_type`: "Purchase Invoice" or "Sales Invoice"

**Returns:** True if all required fields present

**Example:**
```python
from llm_request import validate_invoice_json

if validate_invoice_json(refined_data, "Purchase Invoice"):
    print("JSON is valid")
```

## Integration with Streamlit

The streamlit_app.py now uses this module:

```python
from llm_request import refine_invoice_json_with_llm

# In your Streamlit code
refined_json = refine_invoice_json_with_llm(
    extracted_text=full_text,
    extracted_json=structured_payload,
    invoice_type=invoice_type,
    model_name=llm_model,
    timeout_seconds=llm_timeout
)
```

## Why SDK Instead of REST API?

| Aspect | REST API | SDK |
|--------|----------|-----|
| Authentication | Requires billing setup | Uses .env key |
| Error Handling | Manual JSON parsing | Built-in |
| Fallback | Not available | Automatic SDK fallback |
| Complexity | Higher | Lower |
| Support | Limited | Official |

## Testing

Run the test suite:
```bash
python test_llm_request.py
```

Tests check:
1. ✓ SDK availability
2. ✓ API key loading
3. ✓ Prompt file loading
4. ✓ Sample JSON refinement
5. ✓ Field validation

## Configuration

### Required
- `.env` file with `GEMINI_API_KEY`
- Prompt files in `PDF TO TALLY/prompts/`

### Optional
- Model name (default: gemini-2.5-flash)
- Timeout in seconds (default: 60)

## Troubleshooting

### "No supported Gemini SDK found"
```bash
pip install google-genai google-generativeai
```

### "Missing Gemini API key"
Create `.env` file:
```
GEMINI_API_KEY=your_api_key_here
```

### "Prompt file not found"
Ensure files exist:
- `PDF TO TALLY/prompts/Purchase_Invoice.txt`
- `PDF TO TALLY/prompts/Sales_Invoice.txt`

### "LLM refinement failed"
- Check API key validity
- Verify network connectivity
- Check Gemini API rate limits
- Review error message for details

## Performance

- **Average refine time:** 3-10 seconds per invoice
- **Model:** gemini-2.5-flash recommended
- **Timeout:** 60 seconds (adjustable)
- **Max retries:** None (implement if needed)

## File Structure

```
llm_request.py (main module)
├── SDK detection functions
├── API key loading
├── Prompt management
├── JSON refinement
├── JSON parsing helpers
└── Validation

streamlit_app.py (uses llm_request)
├── Imports refine_invoice_json_with_llm
├── Handles UI/UX
└── Displays results

test_llm_request.py (test suite)
└── Validates all functions
```

## Future Enhancements

- [ ] Retry logic with exponential backoff
- [ ] Batch processing support
- [ ] Custom prompt templates
- [ ] Response caching
- [ ] Async/await support
- [ ] Cost tracking

## License

Same as parent project
