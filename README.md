# Financial Document Analyzer

A single-page web app for analyzing financial PDFs (10-K reports) using Vision Language Models and GPT.

## Features

- Upload and analyze financial PDF documents
- Smart financial table detection with LLM filtering
- Extract key metrics from tables and charts using GPT-4o-mini vision
- Get AI-powered financial analysis with key signals and risk flags
- Interactive chat interface to ask questions about the document

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

3. Run the app:
```bash
streamlit run app.py
```

## Architecture

- **Frontend**: Streamlit
- **PDF Processing**: PyMuPDF with intelligent table detection
- **Table Filtering**: GPT-4o-mini for financial table identification
- **VLM**: GPT-4o-mini vision for metrics extraction from tables
- **LLM**: GPT-4o-mini for financial analysis and chat

## How It Works

1. **Full PDF Scan**: Processes entire document and extracts text from all pages
2. **Two-Stage Filtering**:
   - Structural detection finds pages with any tables
   - LLM filtering identifies only financial tables (filters out TOC, exhibits, etc.)
3. **Vision Extraction**: GPT-4o-mini analyzes table images and extracts metrics
4. **Analysis**: GPT generates summary, key signals, and risk flags
5. **Chat**: Interactive Q&A about the extracted data
