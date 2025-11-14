# Financial Document Analyzer

A single-page web app for analyzing financial PDFs (10-K reports) using Vision Language Models and GPT.

## Features

- Upload and analyze financial PDF documents
- Extract key metrics from tables and charts using DeepSeek VLM
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
# Edit .env and add your API keys
```

3. Run the app:
```bash
streamlit run app.py
```

## Architecture

- **Frontend**: Streamlit
- **PDF Processing**: PyMuPDF
- **VLM**: DeepSeek API for metrics extraction
- **LLM**: OpenAI GPT for analysis and chat
