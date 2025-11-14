"""Simple text chunking for RAG with token counting."""

import re
from typing import List, Dict
import tiktoken
from models import Page


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Count tokens in text using tiktoken.

    Args:
        text: Text to count tokens for
        model: Model name to get the correct encoding

    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base encoding (used by gpt-4 and newer models)
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def detect_section_headers(text: str) -> List[tuple]:
    """
    Detect common 10-K section headers.

    Returns:
        List of (header_text, start_position) tuples
    """
    # Common 10-K section patterns
    section_patterns = [
        r'(?:^|\n)ITEM\s+\d+[A-Z]?\.?\s+[A-Z][^\n]+',  # ITEM 1. Business
        r'(?:^|\n)(?:PART|Part)\s+[IVX]+',              # PART I, PART II, etc.
        r'(?:^|\n)RISK FACTORS',
        r"(?:^|\n)MANAGEMENT[''']?S DISCUSSION",
        r'(?:^|\n)FINANCIAL STATEMENTS',
        r'(?:^|\n)CONSOLIDATED STATEMENTS',
        r'(?:^|\n)NOTES? TO (?:THE )?(?:CONSOLIDATED )?FINANCIAL STATEMENTS',
    ]

    headers = []
    for pattern in section_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            headers.append((match.group().strip(), match.start()))

    # Sort by position
    headers.sort(key=lambda x: x[1])
    return headers


def split_by_sections(text: str, max_chunk_size: int = 500) -> List[str]:
    """
    Split text by section headers, respecting max token size.

    Args:
        text: Full text to split
        max_chunk_size: Maximum tokens per chunk

    Returns:
        List of text chunks
    """
    headers = detect_section_headers(text)

    if not headers:
        # No headers found, fall back to paragraph-based splitting
        return split_by_paragraphs(text, max_chunk_size)

    chunks = []

    # Split text at each header
    for i, (header, start_pos) in enumerate(headers):
        # Get text from this header to the next (or end)
        if i < len(headers) - 1:
            end_pos = headers[i + 1][1]
            section_text = text[start_pos:end_pos]
        else:
            section_text = text[start_pos:]

        # If section is too large, split it further
        section_tokens = count_tokens(section_text)
        if section_tokens > max_chunk_size:
            # Split large sections by paragraphs
            sub_chunks = split_by_paragraphs(section_text, max_chunk_size)
            chunks.extend(sub_chunks)
        else:
            chunks.append(section_text)

    return chunks


def split_by_paragraphs(text: str, max_chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into chunks by paragraphs with overlap.

    Args:
        text: Text to split
        max_chunk_size: Maximum tokens per chunk
        overlap: Number of tokens to overlap between chunks

    Returns:
        List of text chunks
    """
    # Split by double newlines (paragraphs) or single newlines
    paragraphs = re.split(r'\n\s*\n|\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        # If single paragraph exceeds max, split it by sentences
        if para_tokens > max_chunk_size:
            # Save current chunk if any
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_tokens = 0

            # Split large paragraph by sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sentence_chunk = []
            sentence_tokens = 0

            for sentence in sentences:
                sent_tokens = count_tokens(sentence)
                if sentence_tokens + sent_tokens > max_chunk_size and sentence_chunk:
                    chunks.append(' '.join(sentence_chunk))
                    # Keep overlap
                    overlap_text = ' '.join(sentence_chunk[-2:]) if len(sentence_chunk) >= 2 else ''
                    sentence_chunk = [overlap_text, sentence] if overlap_text else [sentence]
                    sentence_tokens = count_tokens(' '.join(sentence_chunk))
                else:
                    sentence_chunk.append(sentence)
                    sentence_tokens += sent_tokens

            if sentence_chunk:
                chunks.append(' '.join(sentence_chunk))
            continue

        # Add paragraph to current chunk
        if current_tokens + para_tokens > max_chunk_size and current_chunk:
            # Save current chunk
            chunks.append('\n\n'.join(current_chunk))

            # Start new chunk with overlap (last paragraph from previous chunk)
            if overlap > 0 and current_chunk:
                overlap_text = current_chunk[-1]
                current_chunk = [overlap_text, para]
                current_tokens = count_tokens('\n\n'.join(current_chunk))
            else:
                current_chunk = [para]
                current_tokens = para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens

    # Add remaining chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


def chunk_text(pages: List[Page], chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """
    Split pages into chunks for RAG retrieval.

    Args:
        pages: List of Page objects from ingestion
        chunk_size: Maximum tokens per chunk (default: 500)
        overlap: Token overlap between chunks (default: 50)

    Returns:
        List of chunk dictionaries with format:
        [{"text": str, "page": int, "chunk_id": int}, ...]
    """
    all_chunks = []
    chunk_id = 0

    for page in pages:
        # Skip pages with no text
        if not page.text or len(page.text.strip()) < 50:
            continue

        # Try section-based splitting first
        page_chunks = split_by_sections(page.text, max_chunk_size=chunk_size)

        # If no sections detected or only one chunk, try paragraph-based
        if len(page_chunks) <= 1:
            page_chunks = split_by_paragraphs(page.text, max_chunk_size=chunk_size, overlap=overlap)

        # Create chunk objects
        for chunk_text in page_chunks:
            # Skip very small chunks (likely noise)
            if len(chunk_text.strip()) < 20:
                continue

            all_chunks.append({
                "text": chunk_text.strip(),
                "page": page.page_number,
                "chunk_id": chunk_id
            })
            chunk_id += 1

    return all_chunks


def get_chunks_by_page(chunks: List[Dict], page_number: int) -> List[Dict]:
    """
    Filter chunks by page number.

    Args:
        chunks: List of chunk dictionaries
        page_number: Page number to filter by

    Returns:
        List of chunks from the specified page
    """
    return [chunk for chunk in chunks if chunk["page"] == page_number]


def preview_chunks(chunks: List[Dict], max_display: int = 5) -> None:
    """
    Print a preview of chunks for debugging.

    Args:
        chunks: List of chunk dictionaries
        max_display: Maximum number of chunks to display
    """
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Previewing first {min(max_display, len(chunks))} chunks:\n")

    for i, chunk in enumerate(chunks[:max_display]):
        tokens = count_tokens(chunk["text"])
        preview_text = chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
        print(f"Chunk {chunk['chunk_id']} (Page {chunk['page']}, {tokens} tokens):")
        print(f"  {preview_text}\n")
