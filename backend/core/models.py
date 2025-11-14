"""Data models for the financial document analyzer."""

from typing import List, Dict, Any
from pydantic import BaseModel


class Metric(BaseModel):
    """Represents a financial metric with values across periods."""
    name: str
    values: List[Dict[str, Any]]  # [{"period": str, "value": float}, ...]

    def to_display_dict(self) -> Dict:
        """Convert to a format suitable for display."""
        return {
            "Metric": self.name,
            **{v["period"]: v["value"] for v in self.values}
        }


class Report(BaseModel):
    """Financial analysis report with summary, signals, and risk flags."""
    summary: str
    key_signals: List[str]
    risk_flags: List[str]


class Page(BaseModel):
    """Represents a single page from a PDF document."""
    page_number: int
    text: str
    image_path: str
    has_tables: bool = False  # Flag to indicate if page contains tables


class ChatMessage(BaseModel):
    """Chat message with role and content."""
    role: str  # "user" or "assistant"
    content: str
