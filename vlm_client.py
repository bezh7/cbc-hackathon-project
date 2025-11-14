"""Vision Language Model client for extracting metrics from document images using GPT-4o-mini."""

import os
import json
import base64
import sys
from typing import List, Tuple
from openai import OpenAI
from models import Page, Metric

# Force unbuffered output
sys.stdout.flush()
sys.stderr.flush()


def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def extract_metrics_with_vlm(pages: List[Page], api_key: str = None) -> Tuple[List[Metric], List[str]]:
    """
    Extract financial metrics from pages using GPT-4o-mini vision model.
    Only processes pages that have tables (has_tables=True).

    Args:
        pages: List of Page objects with text and images
        api_key: OpenAI API key (optional, will use env var if not provided)

    Returns:
        Tuple of (List of Metric objects, List of error messages)
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=api_key)
    all_metrics = []
    errors = []

    # Filter to only pages with tables
    table_pages = [p for p in pages if p.has_tables and p.image_path]

    # Log to file and console
    log_file = "vlm_extraction.log"
    with open(log_file, "a") as f:
        f.write(f"\n=== VLM EXTRACTION DEBUG ===\n")
        f.write(f"Total pages received: {len(pages)}\n")
        f.write(f"Pages with has_tables=True: {len([p for p in pages if p.has_tables])}\n")
        f.write(f"Pages with image_path: {len([p for p in pages if p.image_path])}\n")
        f.write(f"Pages with BOTH (to process): {len(table_pages)}\n")
        if table_pages:
            f.write(f"Page numbers: {[p.page_number for p in table_pages]}\n")

        # Debug: Show first 5 pages details
        f.write(f"\nFirst 5 pages debug:\n")
        for i, page in enumerate(pages[:5]):
            f.write(f"  Page {page.page_number}: has_tables={page.has_tables}, image_path='{page.image_path}'\n")
        f.flush()

    print(f"\n=== VLM EXTRACTION DEBUG ===", flush=True)
    print(f"Total pages received: {len(pages)}", flush=True)
    print(f"Pages with tables to process: {len(table_pages)}", flush=True)
    if table_pages:
        print(f"Page numbers: {[p.page_number for p in table_pages]}", flush=True)

    if not table_pages:
        errors.append("No pages with tables found to process with VLM")
        return all_metrics, errors

    for page in table_pages:
        print(f"\n--- Processing page {page.page_number} ---")
        try:
            # Encode the image
            base64_image = encode_image(page.image_path)

            # Prepare the prompt
            prompt = """Extract all financial metrics from this document page. Look for tables, charts, and text containing financial data.

For each metric you find, extract:
- The metric name (e.g., "Revenue", "Net Income", "Total Assets", "Long-Term Debt")
- The time period (e.g., "2021", "2022", "Q1 2023")
- The value (as a number)

Return ONLY a JSON array in this exact format:
[
  {
    "name": "Revenue",
    "period": "2021",
    "value": 50000
  },
  {
    "name": "Revenue",
    "period": "2022",
    "value": 62000
  }
]

If no financial metrics are found, return an empty array: []

Do not include any explanatory text, only the JSON array."""

            print(f"Image encoded, size: {len(base64_image)} chars")

            # Call GPT-4o-mini vision API
            print("Calling GPT-4o-mini vision API...")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=2000
            )

            # Extract the content
            content = response.choices[0].message.content
            print(f"GPT-4o-mini raw response:\n{content[:500]}...")  # First 500 chars

            # Parse the JSON response
            # Clean up the content to extract just the JSON array
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            print(f"Cleaned content for JSON parsing:\n{content[:300]}...")

            metrics_data = json.loads(content)
            print(f"Parsed {len(metrics_data)} metric entries from JSON")

            # Group metrics by name
            metrics_by_name = {}
            for item in metrics_data:
                name = item.get("name", "Unknown")
                period = item.get("period", "Unknown")
                value = item.get("value", 0)

                if name not in metrics_by_name:
                    metrics_by_name[name] = []

                metrics_by_name[name].append({
                    "period": period,
                    "value": value
                })

            # Create Metric objects
            for name, values in metrics_by_name.items():
                metric = Metric(name=name, values=values)
                all_metrics.append(metric)
                print(f"Created metric: {name} with {len(values)} values")

        except Exception as e:
            error_msg = f"Error processing page {page.page_number}: {str(e)}"
            print(f"ERROR: {error_msg}")
            import traceback
            print(traceback.format_exc())
            errors.append(error_msg)
            # Continue with other pages even if one fails

    print(f"\n=== VLM EXTRACTION COMPLETE ===")
    print(f"Total metrics extracted: {len(all_metrics)}")
    print(f"Metric names: {[m.name for m in all_metrics]}")
    print(f"Errors: {len(errors)}")

    return all_metrics, errors


def merge_duplicate_metrics(metrics: List[Metric]) -> List[Metric]:
    """Merge metrics with the same name from different pages."""
    metrics_by_name = {}

    for metric in metrics:
        if metric.name not in metrics_by_name:
            metrics_by_name[metric.name] = []

        # Add all values, avoiding duplicates
        existing_periods = {v["period"] for v in metrics_by_name[metric.name]}
        for value in metric.values:
            if value["period"] not in existing_periods:
                metrics_by_name[metric.name].append(value)
                existing_periods.add(value["period"])

    # Convert back to Metric objects
    merged_metrics = [
        Metric(name=name, values=values)
        for name, values in metrics_by_name.items()
    ]

    return merged_metrics
