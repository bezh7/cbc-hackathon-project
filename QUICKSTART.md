# Quick Start Guide

This guide will help you get the Financial Document Analyzer up and running in minutes.

## Prerequisites

- Python 3.10 or higher
- OpenAI API key (for vision, analysis, and chat)

## Installation Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=your_actual_openai_key_here
```

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## Using the App

### Step 1: Upload a PDF
- Click "Browse files" in the sidebar
- Select a 10-K or financial PDF document
- The file will be loaded automatically

### Step 2: Analyze the Document
- Click the "Analyze Document" button in the sidebar
- Wait for the 3-step process to complete:
  1. PDF scan and smart financial table detection
  2. Metrics extraction with GPT-4o-mini vision
  3. Financial analysis with GPT

### Step 3: Review the Results
The app will display:
- **Extracted Metrics**: Financial data from tables/charts
- **Key Signals**: Important trends and patterns
- **Risk Flags**: Potential concerns or red flags
- **Summary**: Overall financial situation

### Step 4: Ask Questions
- Use the chat interface at the bottom
- Ask questions about the extracted metrics
- The AI will answer based only on the analyzed document

## Example Questions to Ask

- "What is the revenue growth rate?"
- "Are there any concerning debt levels?"
- "What are the biggest risk factors?"
- "How did net income change year over year?"
- "What trends do you see in the metrics?"

## Troubleshooting

### API Key Errors
- Make sure your `.env` file exists and contains valid keys
- Check that there are no extra spaces around the keys
- Verify your API keys are active and have sufficient credits

### PDF Processing Errors
- Ensure the PDF is not password-protected
- Try a smaller PDF (< 50 pages) for testing
- The app processes only the first 3 pages by default

### VLM Extraction Issues
- Make sure the PDF contains tables or charts with financial data
- Try pages with clear, readable tables
- Higher resolution PDFs work better

### Chat Not Working
- Complete the analysis first before using chat
- Make sure metrics were successfully extracted
- Check that your OpenAI API key is valid

## Tips for Best Results

1. **Choose the right pages**: If you know which pages have financial tables, you can modify the code to target specific pages

2. **Use clear PDFs**: Higher quality PDFs with clear tables produce better results

3. **Be specific in questions**: Ask about specific metrics or time periods for more accurate answers

4. **Check API limits**: Both DeepSeek and OpenAI have rate limits and token limits

## Project Structure

```
.
├── app.py              # Main Streamlit application
├── models.py           # Data models (Metric, Report, Page, etc.)
├── ingestion.py        # PDF processing and image extraction
├── vlm_client.py       # DeepSeek VLM integration
├── llm_client.py       # OpenAI GPT integration
├── config.py           # Configuration settings
├── requirements.txt    # Python dependencies
└── .env               # API keys (create this file)
```

## Next Steps

- Modify `config.py` to adjust processing parameters
- Customize the prompts in `vlm_client.py` and `llm_client.py`
- Add caching to avoid re-processing the same documents
- Extend to support more pages or different document types

## Support

For issues or questions:
1. Check the error messages in the app
2. Review the console output where Streamlit is running
3. Verify API keys and dependencies are correctly installed
