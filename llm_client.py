"""OpenAI GPT client for financial analysis and chat."""

import os
import json
from typing import List, Dict
from openai import OpenAI
from models import Metric, Report, ChatMessage


def get_openai_client(api_key: str = None) -> OpenAI:
    """Get OpenAI client instance."""
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

    return OpenAI(api_key=api_key)


def metrics_to_json_string(metrics: List[Metric]) -> str:
    """Convert metrics list to a compact JSON string."""
    metrics_dict = {}
    for metric in metrics:
        metrics_dict[metric.name] = metric.values

    return json.dumps(metrics_dict, indent=2)


def analyze_metrics_with_gpt(metrics: List[Metric], api_key: str = None) -> Report:
    """
    Analyze financial metrics using GPT to generate insights.

    Args:
        metrics: List of extracted Metric objects
        api_key: OpenAI API key (optional, will use env var if not provided)

    Returns:
        Report object with summary, key signals, and risk flags
    """
    client = get_openai_client(api_key)

    # Convert metrics to JSON format
    metrics_json = metrics_to_json_string(metrics)

    prompt = f"""You are a financial analyst examining financial metrics extracted from a company's regulatory filing.

Here are the extracted metrics:

{metrics_json}

Please analyze these metrics and provide:

1. A brief summary (2-3 sentences) of the overall financial situation
2. 3-5 key signals (important trends, growth patterns, or notable changes)
3. 3-5 risk flags (concerns, declining metrics, or potential red flags)

Format your response as JSON with this exact structure:
{{
  "summary": "Your summary here",
  "key_signals": [
    "Signal 1",
    "Signal 2",
    "Signal 3"
  ],
  "risk_flags": [
    "Risk 1",
    "Risk 2",
    "Risk 3"
  ]
}}

Base your analysis ONLY on the provided metrics. Be specific and reference actual numbers when possible."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a financial analyst. Provide concise, data-driven analysis based only on the metrics provided."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=1000
    )

    content = response.choices[0].message.content

    # Parse the JSON response
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    analysis_data = json.loads(content)

    return Report(
        summary=analysis_data.get("summary", "No summary available"),
        key_signals=analysis_data.get("key_signals", []),
        risk_flags=analysis_data.get("risk_flags", [])
    )


def answer_chat_question(
    question: str,
    metrics: List[Metric],
    report: Report,
    chat_history: List[ChatMessage],
    api_key: str = None
) -> str:
    """
    Answer a user question about the analyzed document using GPT.

    Args:
        question: User's question
        metrics: List of extracted metrics
        report: Analysis report
        chat_history: Previous chat messages for context
        api_key: OpenAI API key (optional, will use env var if not provided)

    Returns:
        Assistant's answer as a string
    """
    client = get_openai_client(api_key)

    # Build context from metrics and report
    metrics_json = metrics_to_json_string(metrics)

    context = f"""You are analyzing a financial document. Here is the extracted information:

METRICS:
{metrics_json}

ANALYSIS SUMMARY:
{report.summary}

KEY SIGNALS:
{chr(10).join(f"- {signal}" for signal in report.key_signals)}

RISK FLAGS:
{chr(10).join(f"- {flag}" for flag in report.risk_flags)}

Answer the user's questions based ONLY on this information. If the information needed to answer is not available in the metrics or analysis, say so clearly. Do not make up or infer information beyond what is provided."""

    # Build messages array
    messages = [
        {
            "role": "system",
            "content": context
        }
    ]

    # Add recent chat history (last 5 messages)
    for msg in chat_history[-5:]:
        messages.append({
            "role": msg.role,
            "content": msg.content
        })

    # Add current question
    messages.append({
        "role": "user",
        "content": question
    })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3,
        max_tokens=800
    )

    return response.choices[0].message.content