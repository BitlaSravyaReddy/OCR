"""
Comprehensive API Key Testing Suite
Tests loading, validation, and working with Gemini API
"""

import json
import os
from pathlib import Path
from llm_request import (
    load_api_key,
    refine_invoice_json_with_llm,
    validate_invoice_json
)


def test_1_env_file_exists():
    """TEST 1: Check if .env file exists"""
    print("\n" + "="*70)
    print("TEST 1: Check if .env file exists")
    print("="*70)
    
    env_path = Path(".env")
    if env_path.exists():
        print("✅ .env file found at:", env_path.absolute())
        content = env_path.read_text()
        # Show if GEMINI_API_KEY line exists (masked)
        for line in content.splitlines():
            if "GEMINI_API_KEY" in line or "GOOGLE_API_KEY" in line:
                key = line.split("=")[0]
                print(f"   ✅ Found key: {key}")
                return True
        print("❌ No GEMINI_API_KEY or GOOGLE_API_KEY found in .env")
        return False
    else:
        print("❌ .env file not found")
        return False


def test_2_env_parsing():
    """TEST 2: Test .env file parsing"""
    print("\n" + "="*70)
    print("TEST 2: Parse .env file directly")
    print("="*70)
    
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ .env file not found")
        return False
    
    content = env_path.read_text(encoding="utf-8")
    print("Raw .env content:")
    print("-" * 70)
    for i, line in enumerate(content.splitlines(), 1):
        if "GEMINI_API_KEY" in line or "GOOGLE_API_KEY" in line:
            # Show masked version
            key_part = line.split("=")[0]
            value_part = "***" + line.split("=")[1][-10:] if "=" in line else "?"
            print(f"Line {i}: {key_part}={value_part}")
        else:
            print(f"Line {i}: {line}")
    
    print("-" * 70)
    
    # Parse like load_api_key does
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() in {"GOOGLE_API_KEY", "GEMINI_API_KEY"}:
            parsed_value = value.strip().strip('"').strip("'")
            print(f"✅ Parsed key '{key.strip()}' with value length: {len(parsed_value)} chars")
            print(f"   First 20 chars: {parsed_value[:20]}")
            print(f"   Last 20 chars:  {parsed_value[-20:]}")
            return len(parsed_value) > 10
    
    print("❌ Could not parse API key from .env")
    return False


def test_3_load_api_key_function():
    """TEST 3: Test load_api_key() function"""
    print("\n" + "="*70)
    print("TEST 3: Test load_api_key() function")
    print("="*70)
    
    # First check environment variables
    print("Step 1: Checking environment variables")
    env_key = os.getenv("GEMINI_API_KEY")
    env_key2 = os.getenv("GOOGLE_API_KEY")
    
    if env_key:
        print(f"  ✅ GEMINI_API_KEY in environment: {len(env_key)} chars")
    else:
        print("  ❌ GEMINI_API_KEY not in environment")
    
    if env_key2:
        print(f"  ✅ GOOGLE_API_KEY in environment: {len(env_key2)} chars")
    else:
        print("  ❌ GOOGLE_API_KEY not in environment")
    
    # Now test load_api_key function
    print("\nStep 2: Calling load_api_key() function")
    api_key = load_api_key()
    
    if api_key:
        print(f"✅ API key loaded successfully")
        print(f"   Length: {len(api_key)} chars")
        print(f"   First 20: {api_key[:20]}")
        print(f"   Last 20:  {api_key[-20:]}")
        
        # Validate format
        if api_key.startswith("AQ."):
            print(f"   ✅ Correct format (starts with AQ.)")
        else:
            print(f"   ⚠️  Unusual format (doesn't start with AQ.)")
        
        return len(api_key) > 30  # Reasonable API key length
    else:
        print("❌ load_api_key() returned empty string")
        return False


def test_4_sdk_availability():
    """TEST 4: Check SDK availability"""
    print("\n" + "="*70)
    print("TEST 4: Check SDK Availability")
    print("="*70)
    
    try:
        from google import genai
        client = genai.Client(api_key="test")
        print("Google GenAI SDK: ✅ Available")
        return True
    except Exception as e:
        print(f"Google GenAI SDK: ❌ Not available - {str(e)}")
        print("Install with: pip install google-genai")
        return False


def test_5_test_api_key_with_simple_request():
    """TEST 5: Test API key by making a simple request"""
    print("\n" + "="*70)
    print("TEST 5: Test API Key with Actual API Call")
    print("="*70)
    
    api_key = load_api_key()
    if not api_key:
        print("❌ No API key available - skipping API call test")
        return False
    
    try:
        print("Attempting to call Gemini API with test prompt...")
        from google import genai
        
        client = genai.Client(api_key=api_key)
        print("✅ Client created successfully with API key")
        
        # Try a very simple request
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Say 'API key works' and nothing else.",
            config={
                "temperature": 0,
                "top_p": 0.95,
                "top_k": 20,
            },
        )
        
        print("✅ API call succeeded!")
        print(f"   Response: {response.text[:100]}")
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ API call failed with error:")
        print(f"   {error_msg}")
        
        # Provide helpful diagnostics
        if "401" in error_msg or "Unauthorized" in error_msg:
            print("\n   💡 This is an authentication error - API key may be invalid")
            print("   Solution: Generate a new API key from https://ai.google.dev/")
        elif "403" in error_msg or "Forbidden" in error_msg:
            print("\n   💡 This is a permission error")
            print("   Solution: Check if API is enabled and project is active")
        elif "429" in error_msg or "quota" in error_msg.lower():
            print("\n   💡 This is a rate limit or quota error")
            print("   Solution: Wait a moment and try again, or check quota")
        elif "No API key" in error_msg:
            print("\n   💡 The SDK is not receiving the API key")
            print("   Solution: Ensure load_api_key() returns a valid key")
        
        return False


def test_6_test_invoice_refinement():
    """TEST 6: Test invoice JSON refinement with real data"""
    print("\n" + "="*70)
    print("TEST 6: Test Invoice JSON Refinement")
    print("="*70)
    
    # Create sample invoice data
    sample_json = {
        "Customer_Name": "John Doe",
        "SupplierName": "ABC Corp",
        "Invoice_Number": "INV-001",
        "Invoice_Date": "2026-04-17",
        "Total_Amount": 1000,
        "SGST": 180,
        "CGST": 0,
        "IGST": 0,
        "productsArray": [
            {"product_name": "Item 1", "quantity": 2, "rate": 500, "amount": 1000}
        ],
        "Expenses": []
    }
    
    sample_text = """
    Invoice INV-001
    Customer: John Doe
    Items:
    - Item 1 x 2 @ 500 = 1000
    Tax: 180 (SGST)
    Total: 1000
    """
    
    try:
        print("Attempting to refine invoice JSON...")
        print(f"Input JSON: {json.dumps(sample_json, indent=2)}")
        
        refined = refine_invoice_json_with_llm(
            extracted_text=sample_text,
            extracted_json=sample_json,
            invoice_type="Purchase Invoice",
            model_name="gemini-2.5-flash",
            timeout_seconds=30
        )
        
        if refined:
            print("✅ Invoice refinement succeeded!")
            print(f"   Refined JSON: {json.dumps(refined, indent=2)[:200]}...")
            
            # Validate
            if validate_invoice_json(refined, "Purchase Invoice"):
                print("✅ Refined JSON has all required fields")
                return True
            else:
                print("⚠️  Refined JSON missing some required fields")
                return False
        else:
            print("❌ Invoice refinement returned None")
            return False
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Invoice refinement failed:")
        print(f"   Error: {error_msg}")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "COMPREHENSIVE API KEY TEST SUITE" + " "*21 + "║")
    print("╚" + "="*68 + "╝")
    
    results = {}
    
    # Run all tests
    results["✅ File Exists"] = test_1_env_file_exists()
    results["✅ Env Parsing"] = test_2_env_parsing()
    results["✅ Load API Key"] = test_3_load_api_key_function()
    results["✅ SDK Available"] = test_4_sdk_availability()
    results["✅ API Call Test"] = test_5_test_api_key_with_simple_request()
    results["✅ Invoice Refine"] = test_6_test_invoice_refinement()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print("-" * 70)
    print(f"Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your API key is working perfectly.")
    elif passed >= 4:
        print("\n⚠️  Most tests passed. Check failing tests for issues.")
    else:
        print("\n❌ Multiple tests failed. Check API key and SDK installation.")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
