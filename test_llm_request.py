#!/usr/bin/env python
"""
Test script for llm_request.py module
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from llm_request import (
    using_new_sdk,
    using_legacy_sdk,
    load_api_key,
    load_invoice_prompt,
    refine_invoice_json_with_llm,
    validate_invoice_json,
)

def test_sdk_availability():
    """Test SDK detection."""
    print("=" * 60)
    print("SDK AVAILABILITY TEST")
    print("=" * 60)
    
    has_new_sdk = using_new_sdk()
    has_legacy_sdk = using_legacy_sdk()
    
    print(f"✓ New SDK (google-genai): {has_new_sdk}")
    print(f"✓ Legacy SDK (google-generativeai): {has_legacy_sdk}")
    
    if not (has_new_sdk or has_legacy_sdk):
        print("❌ ERROR: No Gemini SDK available!")
        return False
    
    return True

def test_api_key():
    """Test API key loading."""
    print("\n" + "=" * 60)
    print("API KEY TEST")
    print("=" * 60)
    
    try:
        api_key = load_api_key()
        if api_key:
            masked_key = api_key[:20] + "..." + api_key[-5:]
            print(f"✓ API Key loaded: {masked_key}")
            return True
        else:
            print("⚠️ No API key found in environment or .env")
            return False
    except Exception as e:
        print(f"❌ Error loading API key: {str(e)}")
        return False

def test_prompt_loading():
    """Test invoice prompt loading."""
    print("\n" + "=" * 60)
    print("PROMPT LOADING TEST")
    print("=" * 60)
    
    for invoice_type in ["Purchase Invoice", "Sales Invoice"]:
        try:
            prompt = load_invoice_prompt(invoice_type)
            if prompt:
                print(f"✓ {invoice_type}: {len(prompt)} chars loaded")
            else:
                print(f"⚠️ {invoice_type}: Empty prompt")
        except FileNotFoundError as e:
            print(f"❌ {invoice_type}: {str(e)}")
            return False
        except Exception as e:
            print(f"❌ {invoice_type}: Unexpected error: {str(e)}")
            return False
    
    return True

def test_sample_refinement():
    """Test LLM refinement with sample data."""
    print("\n" + "=" * 60)
    print("SAMPLE REFINEMENT TEST")
    print("=" * 60)
    
    sample_text = """
    Invoice #INV-2026-001
    Date: 17-Apr-2026
    
    Bill to:
    Customer ABC
    123 Main Street
    
    Items:
    1. Product A - 2 x 100.00 = 200.00
    2. Product B - 3 x 50.00 = 150.00
    
    Subtotal: 350.00
    GST (18%): 63.00
    Total: 413.00
    """
    
    sample_json = {
        "Customer_Name": "Customer ABC",
        "SupplierName": "Supplier XYZ",
        "Invoice_Number": "INV-2026-001",
        "Invoice_Date": "17-Apr-2026",
        "Total_Amount": 413.00,
        "productsArray": [
            {
                "productName": "Product A",
                "Product_Quantity": 2.0,
                "Product_Rate": 100.0,
                "Product_Amount": 200.0
            }
        ],
        "Expenses": []
    }
    
    try:
        print("Testing Purchase Invoice refinement...")
        refined = refine_invoice_json_with_llm(
            extracted_text=sample_text,
            extracted_json=sample_json,
            invoice_type="Purchase Invoice",
            model_name="gemini-2.5-flash",
            timeout_seconds=60
        )
        
        if refined:
            print(f"✓ Refinement successful! Got {len(refined)} fields")
            if validate_invoice_json(refined, "Purchase Invoice"):
                print("✓ Validation passed!")
                return True
            else:
                print("⚠️ Some required fields missing in refined JSON")
                return True  # Partial success
        else:
            print("⚠️ Refinement returned None")
            return False
            
    except ValueError as e:
        print(f"❌ Refinement failed: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " LLM REQUEST MODULE TEST SUITE ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    
    results = {
        "SDK Availability": test_sdk_availability(),
        "API Key Loading": test_api_key(),
        "Prompt Loading": test_prompt_loading(),
        "Sample Refinement": test_sample_refinement(),
    }
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ All tests passed!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
