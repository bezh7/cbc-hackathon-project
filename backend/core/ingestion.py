"""PDF ingestion and processing module."""

import io
import os
from typing import List, Tuple
import fitz  # PyMuPDF
from PIL import Image
from openai import OpenAI
from backend.core.models import Page


def detect_tables(page: fitz.Page) -> bool:
    """
    Detect if a page contains tables using PyMuPDF's table detection.

    Args:
        page: PyMuPDF page object

    Returns:
        True if tables are detected, False otherwise
    """
    try:
        # Try to find tables using PyMuPDF's built-in table detection
        tables = page.find_tables()
        if tables and len(tables.tables) > 0:
            return True

        # Fallback: Check for numeric data in columnar format
        text = page.get_text("text")
        lines = text.split('\n')

        # Count lines with multiple numbers (likely table rows)
        numeric_lines = 0
        for line in lines:
            # Count how many numbers are in this line
            numbers = sum(c.isdigit() or c == '.' or c == ',' for c in line)
            # If line has significant numeric content, count it
            if numbers > 5 and len(line) > 10:
                numeric_lines += 1

        # If we have multiple lines with numbers, likely a table
        return numeric_lines >= 3

    except Exception as e:
        # If table detection fails, be conservative and assume there might be tables
        print(f"Table detection error: {e}")
        return True


def has_financial_tables(page_text: str, page_number: int, api_key: str = None) -> bool:
    """
    Use LLM to determine if a page contains FINANCIAL tables (not just any tables).
    This filters out exhibits, references, appendices, etc.

    Args:
        page_text: Extracted text from the page
        page_number: Page number for context
        api_key: OpenAI API key (optional)

    Returns:
        True if page contains financial metrics/tables, False otherwise
    """
    if not page_text or len(page_text.strip()) < 50:
        return False

    # Skip obviously non-financial pages by keywords
    lower_text = page_text.lower()
    skip_keywords = [
        'table of contents',
        'index to exhibits',
        'exhibit index',
        'signature',
        'power of attorney',
        'form 10-k index',
    ]

    for keyword in skip_keywords:
        if keyword in lower_text:
            return False

    # Get API key
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        # If no API key, fall back to basic keyword matching
        return _basic_financial_detection(page_text)

    try:
        client = OpenAI(api_key=api_key)

        # Truncate text to first 1000 chars to save tokens
        sample_text = page_text[:1000]

        prompt = f"""Analyze this page excerpt from a 10-K filing and determine if it contains FINANCIAL TABLES with numeric metrics.

Page excerpt:
{sample_text}

Return ONLY "YES" if this page contains:
- Financial statements (balance sheet, income statement, cash flow)
- Tables with financial metrics (revenue, expenses, assets, liabilities, etc.)
- Numeric financial data in table format

Return ONLY "NO" if this page contains:
- Table of contents or indexes
- Exhibits list or references
- Legal disclaimers
- Text-only content
- Non-financial tables

Answer (YES or NO):"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cheap and fast
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial document analyst. Respond with only YES or NO."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=5
        )

        answer = response.choices[0].message.content.strip().upper()
        return "YES" in answer

    except Exception as e:
        print(f"LLM financial detection error on page {page_number}: {e}")
        # Fall back to basic detection if LLM fails
        return _basic_financial_detection(page_text)


def _basic_financial_detection(text: str) -> bool:
    """Fallback financial table detection using keywords."""
    financial_keywords = [
        'revenue', 'income', 'expense', 'asset', 'liability',
        'stockholders equity', 'cash flow', 'operating income',
        'net income', 'total assets', 'balance sheet',
        'earnings per share', 'eps', 'comprehensive income'
    ]

    lower_text = text.lower()
    keyword_count = sum(1 for keyword in financial_keywords if keyword in lower_text)

    # If we find multiple financial keywords, likely a financial table
    return keyword_count >= 2


def ingest_pdf(pdf_bytes: bytes, max_pages: int = None, output_dir: str = "temp_images", use_llm_filter: bool = True) -> Tuple[List[Page], dict]:
    """
    Ingest a PDF file with two-stage table detection:
    1. Structural detection (find pages with any tables)
    2. LLM filtering (identify pages with FINANCIAL tables only)

    Args:
        pdf_bytes: Raw PDF file bytes
        max_pages: Maximum number of pages to process (None = all pages)
        output_dir: Directory to save page images
        use_llm_filter: Whether to use LLM for financial table detection (default: True)

    Returns:
        Tuple of (List of Page objects, stats dict with processing info)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open PDF from bytes
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = []
    total_pages = len(pdf_document)
    num_pages = total_pages if max_pages is None else min(total_pages, max_pages)

    pages_with_any_tables = 0
    pages_with_financial_tables = 0
    pages_without_tables = 0

    # Get API key for LLM filtering
    api_key = os.getenv("OPENAI_API_KEY") if use_llm_filter else None

    for page_num in range(num_pages):
        page = pdf_document[page_num]

        # Extract text from every page
        text = page.get_text("text")

        # Stage 1: Structural table detection (fast, broad)
        has_any_tables = detect_tables(page)

        # Stage 2: Financial table detection (LLM-based, precise)
        has_financial_table = False
        if has_any_tables and use_llm_filter:
            pages_with_any_tables += 1
            has_financial_table = has_financial_tables(text, page_num + 1, api_key)
        elif has_any_tables:
            # If not using LLM filter, treat any table as financial
            has_financial_table = True
            pages_with_any_tables += 1

        if has_financial_table:
            pages_with_financial_tables += 1
            # Only render image for pages with financial tables
            zoom = 2.0  # zoom factor for higher resolution
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            # Save image
            image_filename = f"page_{page_num + 1}.png"
            image_path = os.path.join(output_dir, image_filename)
            pix.save(image_path)
        else:
            pages_without_tables += 1
            # No image needed for non-financial pages
            image_path = ""

        # Create Page object - mark as has_tables only if it has FINANCIAL tables
        page_obj = Page(
            page_number=page_num + 1,
            text=text,
            image_path=image_path,
            has_tables=has_financial_table
        )
        pages.append(page_obj)

    pdf_document.close()

    # Return stats for UI display
    stats = {
        "total_pages": total_pages,
        "processed_pages": num_pages,
        "pages_with_tables": pages_with_financial_tables,
        "pages_without_tables": pages_without_tables,
        "pages_filtered_out": pages_with_any_tables - pages_with_financial_tables if use_llm_filter else 0
    }

    return pages, stats


def cleanup_images(output_dir: str = "temp_images"):
    """Clean up temporary image files."""
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        try:
            os.rmdir(output_dir)
        except OSError:
            pass  # Directory not empty or other error
