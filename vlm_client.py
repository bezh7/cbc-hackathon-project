"""DeepSeek VLM client for extracting metrics from document images."""

import os
import json
import base64
from typing import List
import requests
from models import Page, Metric


def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def extract_metrics_with_vlm(pages: List[Page], api_key: str = None) -> List[Metric]:
    """
    Extract financial metrics from pages using DeepSeek VLM.

    Args:
        pages: List of Page objects with text and images
        api_key: DeepSeek API key (optional, will use env var if not provided)

    Returns:
        List of Metric objects
    """
    if api_key is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise ValueError("DeepSeek API key not found. Set DEEPSEEK_API_KEY environment variable.")

    all_metrics = []

    for page in pages:
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

            # Call DeepSeek VLM API
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "deepseek-vl",
                "messages": [
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
                "temperature": 0.1,
                "max_tokens": 2000
            }

            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )

            response.raise_for_status()
            result = response.json()

            # Extract the content
            content = result["choices"][0]["message"]["content"]

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

            metrics_data = json.loads(content)

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

        except Exception as e:
            print(f"Error processing page {page.page_number}: {str(e)}")
            # Continue with other pages even if one fails

    return all_metrics


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
