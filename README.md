# OCR Document Extractor

A Streamlit web application for extracting text and structured data from PDF and image documents using advanced OCR and AI processing.

## Features

- 📄 **Multi-format support**: PDF, PNG, JPG, JPEG, BMP, TIFF, WebP
- 🌍 **Multi-language OCR**: English, Hindi, and other languages
- 🤖 **AI-powered refinement**: Optional Google Gemini integration for better data structuring
- 📊 **Structured output**: Extracts key-value pairs, tables, and normalized fields
- 💾 **Download results**: Get both raw text and structured JSON outputs

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API key** (optional, for AI enhancement):
   - Get a Google Gemini API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
   - Create a `.env` file in the project root:
     ```
     GEMINI_API_KEY=your_api_key_here
     ```

## Usage

### Running the Web App

```bash
streamlit run streamlit_app.py
```

This will start the web application at `http://localhost:8501`

### Configuration Options

- **OCR Languages**: Comma-separated list (e.g., "en,hi,es")
- **PDF DPI**: Rendering quality for PDFs (higher = better but slower)
- **Token Confidence**: Minimum OCR confidence threshold
- **AI Enhancement**: Enable/disable Gemini AI refinement
- **AI Model**: Choose Gemini model (gemini-2.5-flash, gemini-2.0-flash, etc.)
- **AI Timeout**: Maximum wait time for AI processing

### Command Line Usage (OCR Pipeline)

You can also use the underlying OCR pipeline directly:

```bash
python ocr_pipeline.py --input-dir Completed --output-dir results --languages en,hi
```

## How It Works

1. **Upload Files**: Select one or more PDF/image files
2. **OCR Processing**: Uses PaddleOCR with multi-language support
3. **Text Extraction**: Extracts raw text from documents
4. **Structure Analysis**: Identifies layout, tables, and key-value pairs
5. **AI Refinement** (optional): Uses Google Gemini to improve data organization
6. **Results Display**: Shows extracted text, metadata, and structured JSON

## Output Structure

The application generates:

- **Raw Text**: Full extracted text content
- **Structured JSON**: Contains:
  - `normalized_fields`: Key-value pairs (invoice_no, date, total_amount, etc.)
  - `records`: Table data extracted from documents
  - `layout`: Word-level coordinates and layout information
  - `rows`: Y-coordinate grouped rows
  - `pages`: Per-page analysis
  - `llm_analysis`: AI refinement results

## Supported Document Types

- **Invoices**: Purchase invoices, sales invoices, bills
- **Receipts**: Payment receipts, expense receipts
- **Forms**: Structured documents with fields and tables
- **Reports**: Financial reports, statements

## Requirements

- Python 3.8+
- Internet connection (for AI features)
- Google Gemini API key (optional, for AI enhancement)

## Troubleshooting

### Common Issues

1. **"No module named 'paddleocr'"**:
   ```bash
   pip install paddlepaddle paddleocr
   ```

2. **"Authentication error"**:
   - Check your `.env` file has the correct `GEMINI_API_KEY`
   - Ensure the API key is active at aistudio.google.com/apikey

3. **"Quota exceeded"**:
   - Free tier has daily limits
   - Create a new project/key or wait for quota reset

4. **Slow processing**:
   - Reduce PDF DPI setting
   - Disable AI enhancement for faster processing

### Performance Tips

- Use lower DPI (150-200) for faster PDF processing
- Disable AI enhancement if you only need raw text
- Process files in batches rather than all at once

## License

This project is for educational and commercial use.