"""PDF ingestion and processing module."""

import io
import os
from typing import List
import fitz  # PyMuPDF
from PIL import Image
from models import Page


def ingest_pdf(pdf_bytes: bytes, max_pages: int = 3, output_dir: str = "temp_images") -> List[Page]:
    """
    Ingest a PDF file and extract text and images from selected pages.

    Args:
        pdf_bytes: Raw PDF file bytes
        max_pages: Maximum number of pages to process (default: 3)
        output_dir: Directory to save page images

    Returns:
        List of Page objects with extracted text and image paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open PDF from bytes
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

    pages = []
    num_pages = min(len(pdf_document), max_pages)

    for page_num in range(num_pages):
        page = pdf_document[page_num]

        # Extract text
        text = page.get_text("text")

        # Render page to image
        # Use higher resolution for better VLM processing
        zoom = 2.0  # zoom factor for higher resolution
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # Save image
        image_filename = f"page_{page_num + 1}.png"
        image_path = os.path.join(output_dir, image_filename)
        pix.save(image_path)

        # Create Page object
        page_obj = Page(
            page_number=page_num + 1,
            text=text,
            image_path=image_path
        )
        pages.append(page_obj)

    pdf_document.close()

    return pages


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
