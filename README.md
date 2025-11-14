# 📊 Financial Document Analyzer

An AI-powered tool for analyzing 10-K financial filings using GPT-4o-mini Vision, RAG (Retrieval Augmented Generation), and MongoDB persistence.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

- **🔍 Smart Table Detection** - LLM-powered filtering to identify only financial tables (filters out TOC, exhibits, etc.)
- **👁️ Vision-Based Metric Extraction** - GPT-4o-mini Vision reads complex tables and charts
- **🤖 AI Financial Analysis** - Automated generation of insights, trends, and risk flags
- **💬 RAG-Powered Chat** - Ask questions about the entire document with source citations
- **📊 MongoDB Persistence** - Store and query structured financial data across multiple filings
- **📈 Time Series Support** - Track metrics across multiple years and quarters

## 🏗️ Architecture

```
├── frontend/              # Streamlit UI
│   └── app.py            # Main application
├── backend/
│   ├── core/             # Core business logic
│   │   ├── models.py     # Data models (Pydantic)
│   │   ├── ingestion.py  # PDF processing & table detection
│   │   ├── vlm_client.py # GPT-4o-mini Vision extraction
│   │   └── llm_analysis.py # GPT analysis & RAG chat
│   ├── rag/              # RAG system
│   │   ├── rag_chunking.py  # Text chunking with token counting
│   │   └── rag_retrieval.py # In-memory vector search
│   └── database/         # MongoDB integration
│       ├── mongo_client.py  # Connection helper
│       └── persistence.py   # CRUD operations
├── docs/                 # Documentation
└── requirements.txt      # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key
- MongoDB Atlas account (free tier works)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/financial-document-analyzer.git
   cd financial-document-analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

4. **Run the application**
   ```bash
   streamlit run frontend/app.py
   ```

5. **Open in browser**
   - Navigate to `http://localhost:8501`

## 📝 Configuration

Create a `.env` file in the project root:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# MongoDB (optional - for persistence)
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
MONGODB_DB_NAME=financial_analyzer
```

## 💻 Usage

### 1. Upload & Analyze
- Upload a 10-K PDF
- Enter company information (name, ticker, fiscal year)
- Click "Analyze Document"

### 2. View Results
- **Extracted Metrics** - Revenue, Net Income, Assets, etc. with multi-period values
- **Key Signals** - Important trends and growth patterns
- **Risk Flags** - Concerns and red flags identified by AI
- **Summary** - Overall financial situation

### 3. Chat with Document
- Ask questions like:
  - "What are the main risk factors?"
  - "How did revenue change year-over-year?"
  - "What does management say about future outlook?"
- Get answers with **source page citations**

### 4. Save to Database (Optional)
- Click "Save to Database" to persist metrics
- View recent filings and metrics in the database preview

## 🔧 How It Works

### Analysis Pipeline

```
PDF Upload
    ↓
[1. Ingestion] → Smart table detection with LLM filtering
    ↓
[2. Vision Extraction] → GPT-4o-mini reads tables → Structured metrics
    ↓
[3. Text Chunking] → Section-aware chunking (~500 tokens)
    ↓
[4. Embedding] → OpenAI embeddings (text-embedding-3-small)
    ↓
[5. Analysis] → GPT generates insights
    ↓
Results + RAG Chat
```

### Chat/Query Flow

```
User Question
    ↓
[1. Embed Query] → text-embedding-3-small
    ↓
[2. Semantic Search] → Cosine similarity (top-2 chunks)
    ↓
[3. Context Building] → Metrics + Chunks + Report
    ↓
[4. LLM Response] → GPT-4o-mini with grounded context
    ↓
Answer + Source Pages
```

## 📊 Database Schema

### Collections

**filings**
```json
{
  "filing_id": "uuid",
  "company": "Apple Inc.",
  "ticker": "AAPL",
  "year": 2023,
  "filename": "AAPL_10K_2023.pdf",
  "uploaded_at": "2024-01-15T10:30:00Z"
}
```

**metrics**
```json
{
  "metric_id": "uuid",
  "filing_id": "uuid",
  "name": "Revenue",
  "period": "2023",
  "value": 394328000000,
  "source_pages": [42, 43],
  "extracted_at": "2024-01-15T10:35:00Z"
}
```

## 💰 Cost Estimates

Per document analysis (~200 pages):
- PDF ingestion: Free
- VLM extraction (3-5 pages): ~$0.01
- Text chunking: Free
- Embeddings (~100 chunks): ~$0.0002
- GPT analysis: ~$0.001
- **Total: ~$0.011 per document**

Per chat query: ~$0.001

## 🛠️ Development

### Project Structure

- `frontend/` - Streamlit UI layer
- `backend/core/` - Business logic (ingestion, extraction, analysis)
- `backend/rag/` - RAG implementation (chunking, retrieval)
- `backend/database/` - MongoDB persistence layer
- `docs/` - Documentation and guides

### Running Tests

```bash
python -m pytest tests/
```

### Code Style

```bash
# Format code
black .

# Lint
flake8 .
```

## 🐛 Troubleshooting

### MongoDB Connection Issues
- Ensure your IP is whitelisted in MongoDB Atlas Network Access
- Check connection string format
- Verify username/password

### Rate Limit Errors
- The app has automatic retry logic with exponential backoff
- If persistent, upgrade your OpenAI plan or reduce batch sizes

### Import Errors
- Make sure you're running from the project root
- Check that all dependencies are installed: `pip install -r requirements.txt`

## 📚 Documentation

- [RAG Implementation Guide](RAG_IMPLEMENTATION.md)
- [Database Schema](docs/database-schema.md)
- [API Documentation](docs/api.md)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT-4o-mini and embeddings API
- Streamlit for the amazing UI framework
- PyMuPDF for PDF processing capabilities
