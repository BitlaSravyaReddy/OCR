import streamlit as st
import tempfile
import shutil
from pathlib import Path
import json
from typing import List, Tuple, Optional, Dict, Any
import sys
import os

# Add current directory to path so we can import modules
sys.path.insert(0, os.path.dirname(__file__))

from ocr_pipeline import (
    collect_input_files,
    HybridOCR,
    StructuredExtractor,
    GeminiLayoutAnalyzer,
    write_output_text,
    write_output_json,
    expected_output_text_path,
    expected_output_json_path,
    FileSummary,
    parse_args as ocr_parse_args,
    main as ocr_main
)

from llm_request import (
    refine_invoice_json_with_llm,
    load_api_key,
    using_new_sdk,
    using_legacy_sdk,
)

st.set_page_config(
    page_title="OCR Document Extractor",
    page_icon="📄",
    layout="wide"
)

st.title("📄 OCR Invoice Extractor")
st.markdown("Upload PDF or image files to extract and structure invoice data using advanced OCR and AI processing.")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# OCR Languages
languages_input = st.sidebar.text_input(
    "OCR Languages",
    value="en,hi",
    help="Comma-separated list of languages (e.g., en,hi,es)"
)

# PDF DPI
pdf_dpi = st.sidebar.slider(
    "PDF Render DPI",
    min_value=150,
    max_value=600,
    value=300,
    step=50,
    help="Higher DPI = better quality but slower processing"
)

# Minimum token confidence
min_confidence = st.sidebar.slider(
    "Minimum Token Confidence",
    min_value=0.1,
    max_value=0.9,
    value=0.20,
    step=0.05,
    help="Lower values include more OCR results but may be less accurate"
)

# LLM Configuration
st.sidebar.subheader("🤖 AI Enhancement")
use_llm = st.sidebar.checkbox(
    "Enable AI-powered JSON refinement",
    value=True,
    help="Use Gemini AI to generate structured invoice JSON"
)

llm_model = st.sidebar.selectbox(
    "AI Model",
    options=["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-pro"],
    index=0,
    help="Choose the AI model for data refinement"
)

llm_timeout = st.sidebar.slider(
    "AI Timeout (seconds)",
    min_value=30,
    max_value=300,
    value=60,
    step=30,
    help="Maximum time to wait for AI processing"
)

# File uploader
st.header("📤 Upload Invoice Files")

# Invoice type selection
col1, col2 = st.columns(2)
with col1:
    invoice_type = st.radio(
        "Select Invoice Type:",
        options=["Purchase Invoice", "Sales Invoice"],
        horizontal=True
    )

uploaded_files = st.file_uploader(
    "Choose PDF or image files",
    type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff", "webp"],
    accept_multiple_files=True,
    help="Upload one or more invoice documents to process"
)

def load_invoice_prompt(invoice_type: str) -> str:
    """Load the appropriate invoice prompt based on type."""
    from llm_request import load_invoice_prompt as get_prompt
    return get_prompt(invoice_type)

if uploaded_files:
    st.success(f"📁 {len(uploaded_files)} file(s) uploaded successfully!")

    # Process button
    if st.button("🚀 Extract & Refine Invoice Data", type="primary", use_container_width=True):
        with st.spinner("🔄 Processing invoices... This may take a few minutes."):

            # Create temporary directories
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_input_dir = Path(temp_dir) / "input"
                temp_output_dir = Path(temp_dir) / "output"

                temp_input_dir.mkdir(parents=True, exist_ok=True)
                temp_output_dir.mkdir(parents=True, exist_ok=True)

                # Save uploaded files to temp input directory
                saved_files = []
                for uploaded_file in uploaded_files:
                    file_path = temp_input_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_files.append(file_path)

                # Prepare OCR arguments
                languages = [lang.strip() for lang in languages_input.split(",") if lang.strip()]

                # Initialize components
                extractor = HybridOCR(
                    languages=languages,
                    render_dpi=pdf_dpi,
                    min_token_confidence=min_confidence,
                )
                structurer = StructuredExtractor()

                # Process each file
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, file_path in enumerate(saved_files):
                    status_text.text(f"Processing: {file_path.name}")
                    progress_bar.progress(i / len(saved_files))

                    try:
                        # Process file with OCR
                        if file_path.suffix.lower() in [".pdf"]:
                            page_results = extractor.process_pdf_pages(file_path)
                        else:
                            page_results = extractor.process_image_pages(file_path)

                        full_text = "\n\n".join(page.text for page in page_results if page.text)
                        sources = [page.source for page in page_results]

                        # Extract structure
                        structured_payload = structurer.extract_document_structure(file_path, page_results)
                        
                        # Refine with LLM using invoice-specific prompts
                        refined_json = None
                        llm_status = "disabled"
                        
                        if use_llm:
                            status_text.text(f"Refining with AI: {file_path.name}")
                            try:
                                refined_json = refine_invoice_json_with_llm(
                                    extracted_text=full_text,
                                    extracted_json=structured_payload,
                                    invoice_type=invoice_type,
                                    model_name=llm_model,
                                    timeout_seconds=llm_timeout
                                )
                                llm_status = "success" if refined_json else "failed"
                            except ValueError as e:
                                st.warning(f"⚠️ AI refinement warning for {file_path.name}: {str(e)}")
                                llm_status = "failed"
                                refined_json = None
                            except Exception as e:
                                st.error(f"❌ AI refinement error for {file_path.name}: {str(e)}")
                                llm_status = "error"
                                refined_json = None

                        results.append({
                            "file_name": file_path.name,
                            "status": "success",
                            "text": full_text,
                            "raw_json": structured_payload,
                            "refined_json": refined_json,
                            "pages": len(page_results),
                            "sources": sources,
                            "orientation": structured_payload.get("document_orientation", "unknown"),
                            "llm_status": llm_status,
                            "invoice_type": invoice_type
                        })

                    except Exception as e:
                        results.append({
                            "file_name": file_path.name,
                            "status": "error",
                            "error": str(e)
                        })

                progress_bar.progress(1.0)
                status_text.text("✅ Processing complete!")

        # Display results
        st.header("📋 Extraction Results")

        # Summary
        success_count = sum(1 for r in results if r["status"] == "success")
        error_count = sum(1 for r in results if r["status"] == "error")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("✅ Successful", success_count)
        with col2:
            st.metric("❌ Failed", error_count)
        with col3:
            st.metric("📄 Total Files", len(results))

        # Individual file results
        for result in results:
            with st.expander(f"📄 {result['file_name']} ({result.get('invoice_type', 'Unknown')})", expanded=True):
                if result["status"] == "success":
                    
                    # Tabs for different views
                    tab1, tab2, tab3 = st.tabs(["📊 Refined JSON", "🔍 Raw Extraction", "📝 Raw Text"])
                    
                    with tab1:
                        if result["refined_json"]:
                            st.subheader("✨ AI-Refined Invoice Data")
                            st.success(f"AI Enhancement: {result['llm_status'].upper()}")
                            
                            # Display refined JSON with better formatting
                            refined_json_str = json.dumps(result["refined_json"], indent=2, ensure_ascii=False)
                            st.code(refined_json_str, language="json")
                            
                            # Download button for refined JSON
                            st.download_button(
                                "📥 Download Refined JSON",
                                refined_json_str,
                                file_name=f"{Path(result['file_name']).stem}_refined.json",
                                mime="application/json",
                                key=f"download_refined_{result['file_name']}"
                            )
                        else:
                            st.warning("⚠️ No refined JSON available. Using raw extraction instead.")
                            raw_json_str = json.dumps(result["raw_json"], indent=2, ensure_ascii=False)
                            st.code(raw_json_str, language="json")
                    
                    with tab2:
                        st.subheader("🔧 Raw OCR Extraction")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write(f"**Pages:** {result['pages']}")
                            st.write(f"**Orientation:** {result['orientation']}")
                            st.write(f"**Sources:** {', '.join(result['sources'])}")

                        with col2:
                            st.write(f"**Invoice Type:** {result['invoice_type']}")
                            st.write(f"**AI Status:** {result['llm_status']}")

                        # Raw JSON
                        raw_json_str = json.dumps(result["raw_json"], indent=2, ensure_ascii=False)
                        st.code(raw_json_str, language="json")
                        
                        st.download_button(
                            "📥 Download Raw JSON",
                            raw_json_str,
                            file_name=f"{Path(result['file_name']).stem}_raw.json",
                            mime="application/json",
                            key=f"download_raw_{result['file_name']}"
                        )
                    
                    with tab3:
                        st.subheader("📄 Extracted Text")
                        st.text_area(
                            "Full extracted text content",
                            result["text"],
                            height=300,
                            key=f"text_{result['file_name']}"
                        )
                        
                        st.download_button(
                            "📥 Download Text",
                            result["text"],
                            file_name=f"{Path(result['file_name']).stem}.txt",
                            mime="text/plain",
                            key=f"download_text_{result['file_name']}"
                        )

                else:
                    st.error(f"❌ Processing failed: {result['error']}")

else:
    st.info("👆 Upload one or more PDF or image files to get started!")

# Footer
st.markdown("---")
st.markdown(
    """
    **About this tool:**
    - 🤖 Advanced OCR with multi-language support (English, Hindi, etc.)
    - 🏢 Invoice-specific data extraction for Purchase and Sales invoices
    - ✨ AI-powered JSON refinement using Google Gemini
    - 📋 Structured output with product details, customer info, and tax breakdowns
    - 💾 Download results as JSON or text
    
    **Supported Formats:** PDF, PNG, JPG, JPEG, BMP, TIFF, WebP
    """
)



if uploaded_files:
    st.success(f"📁 {len(uploaded_files)} file(s) uploaded successfully!")

    # Process button
    if st.button("🚀 Extract Content", type="primary", use_container_width=True):
        with st.spinner("🔄 Processing documents... This may take a few minutes."):

            # Create temporary directories
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_input_dir = Path(temp_dir) / "input"
                temp_output_dir = Path(temp_dir) / "output"

                temp_input_dir.mkdir(parents=True, exist_ok=True)
                temp_output_dir.mkdir(parents=True, exist_ok=True)

                # Save uploaded files to temp input directory
                saved_files = []
                for uploaded_file in uploaded_files:
                    file_path = temp_input_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_files.append(file_path)

                # Prepare OCR arguments
                languages = [lang.strip() for lang in languages_input.split(",") if lang.strip()]

                # Initialize components
                extractor = HybridOCR(
                    languages=languages,
                    render_dpi=pdf_dpi,
                    min_token_confidence=min_confidence,
                )
                structurer = StructuredExtractor()
                llm_analyzer = GeminiLayoutAnalyzer(
                    model=llm_model,
                    timeout_seconds=llm_timeout,
                    enabled=use_llm,
                )

                # Process each file
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, file_path in enumerate(saved_files):
                    status_text.text(f"Processing: {file_path.name}")
                    progress_bar.progress((i) / len(saved_files))

                    try:
                        # Process file
                        if file_path.suffix.lower() in [".pdf"]:
                            page_results = extractor.process_pdf_pages(file_path)
                        else:
                            page_results = extractor.process_image_pages(file_path)

                        full_text = "\n\n".join(page.text for page in page_results if page.text)
                        sources = [page.source for page in page_results]

                        # Extract structure
                        structured_payload = structurer.extract_document_structure(file_path, page_results)

                        # LLM refinement
                        llm_result = llm_analyzer.refine(structured_payload)
                        structured_payload["llm_analysis"] = llm_result

                        if llm_result.get("status") == "success":
                            analysis = llm_result.get("analysis", {})
                            if "normalized_fields" in analysis:
                                structured_payload["normalized_fields_refined"] = analysis["normalized_fields"]
                            if "records" in analysis:
                                structured_payload["records_refined"] = analysis["records"]

                        structured_payload["final_fields"] = structured_payload.get(
                            "normalized_fields_refined",
                            structured_payload.get("normalized_fields", {}),
                        )
                        structured_payload["final_records"] = structured_payload.get(
                            "records_refined",
                            structured_payload.get("records", []),
                        )

                        # Write outputs
                        output_txt = write_output_text(temp_output_dir, file_path, full_text)
                        output_json = write_output_json(temp_output_dir, file_path, structured_payload)

                        results.append({
                            "file_name": file_path.name,
                            "status": "success",
                            "text": full_text,
                            "json_data": structured_payload,
                            "pages": len(page_results),
                            "sources": sources,
                            "orientation": structured_payload.get("document_orientation", "unknown"),
                            "llm_status": llm_result.get("status", "not_run")
                        })

                    except Exception as e:
                        results.append({
                            "file_name": file_path.name,
                            "status": "error",
                            "error": str(e)
                        })

                progress_bar.progress(1.0)
                status_text.text("✅ Processing complete!")

        # Display results
        st.header("📋 Results")

        # Summary
        success_count = sum(1 for r in results if r["status"] == "success")
        error_count = sum(1 for r in results if r["status"] == "error")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("✅ Successful", success_count)
        with col2:
            st.metric("❌ Failed", error_count)
        with col3:
            st.metric("📄 Total Files", len(results))

        # Individual file results
        for result in results:
            with st.expander(f"📄 {result['file_name']}", expanded=True):
                if result["status"] == "success":
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("📊 Metadata")
                        st.write(f"**Pages:** {result['pages']}")
                        st.write(f"**Orientation:** {result['orientation']}")
                        st.write(f"**AI Enhancement:** {result['llm_status']}")
                        st.write(f"**Sources:** {', '.join(result['sources'])}")

                    with col2:
                        st.subheader("🔍 Extracted Fields")
                        final_fields = result["json_data"].get("final_fields", {})
                        if final_fields:
                            for field, value in final_fields.items():
                                st.write(f"**{field.replace('_', ' ').title()}:** {value}")
                        else:
                            st.write("No structured fields extracted")

                    # Raw text
                    st.subheader("📝 Raw Text")
                    st.text_area(
                        "Extracted text content",
                        result["text"],
                        height=200,
                        key=f"text_{result['file_name']}"
                    )

                    # JSON data
                    st.subheader("📋 Structured JSON")
                    json_str = json.dumps(result["json_data"], indent=2, ensure_ascii=False)
                    st.code(json_str, language="json")

                    # Download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "📥 Download Text",
                            result["text"],
                            file_name=f"{result['file_name']}.txt",
                            mime="text/plain",
                            key=f"download_text_{result['file_name']}"
                        )
                    with col2:
                        st.download_button(
                            "📥 Download JSON",
                            json_str,
                            file_name=f"{result['file_name']}.json",
                            mime="application/json",
                            key=f"download_json_{result['file_name']}"
                        )

                else:
                    st.error(f"❌ Processing failed: {result['error']}")

else:
    st.info("👆 Upload one or more PDF or image files to get started!")

# Footer
st.markdown("---")
st.markdown(
    """
    **About this tool:**
    - Uses advanced OCR with multi-language support (English, Hindi, etc.)
    - Extracts structured data from invoices, receipts, and documents
    - Optional AI enhancement using Google Gemini for better data organization
    - Supports PDF and various image formats
    """
)