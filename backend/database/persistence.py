"""Functions for saving financial data to MongoDB."""

import uuid
from datetime import datetime
from typing import List, Dict
from backend.core.models import Metric
from backend.database.mongo_client import get_db


def create_filing_record(
    company: str,
    ticker: str,
    year: int,
    filename: str
) -> str:
    """
    Create a new filing record in MongoDB.

    Args:
        company: Company name (e.g., "Apple Inc.")
        ticker: Stock ticker (e.g., "AAPL")
        year: Fiscal year (e.g., 2023)
        filename: PDF filename

    Returns:
        filing_id (UUID string)
    """
    db = get_db()
    filings_collection = db.filings

    # Generate unique filing ID
    filing_id = str(uuid.uuid4())

    # Create filing document
    filing_doc = {
        "filing_id": filing_id,
        "company": company,
        "ticker": ticker,
        "year": year,
        "filename": filename,
        "uploaded_at": datetime.utcnow()
    }

    # Insert into MongoDB
    filings_collection.insert_one(filing_doc)

    return filing_id


def save_metrics(
    filing_id: str,
    metrics: List[Metric],
    source_pages: Dict[str, List[int]] = None
) -> int:
    """
    Save extracted metrics to MongoDB.

    Each Metric object contains a name and list of period/value pairs.
    This function creates one document per period/value pair.

    Args:
        filing_id: UUID of the filing record
        metrics: List of Metric objects
        source_pages: Optional dict mapping metric names to page numbers

    Returns:
        Number of metric documents saved
    """
    db = get_db()
    metrics_collection = db.metrics

    if source_pages is None:
        source_pages = {}

    metric_docs = []

    # Create one document per metric period/value pair
    for metric in metrics:
        for value_data in metric.values:
            metric_doc = {
                "metric_id": str(uuid.uuid4()),
                "filing_id": filing_id,
                "name": metric.name,
                "period": value_data.get("period"),
                "value": value_data.get("value"),
                "source_pages": source_pages.get(metric.name, []),
                "extracted_at": datetime.utcnow()
            }
            metric_docs.append(metric_doc)

    # Insert all metrics at once
    if metric_docs:
        metrics_collection.insert_many(metric_docs)

    return len(metric_docs)


def get_recent_filings(limit: int = 5) -> List[Dict]:
    """
    Get the most recent filing records.

    Args:
        limit: Maximum number of records to return

    Returns:
        List of filing documents (as dicts)
    """
    db = get_db()
    filings_collection = db.filings

    # Get recent filings, sorted by upload date
    filings = list(
        filings_collection
        .find({}, {"_id": 0})  # Exclude MongoDB's internal _id
        .sort("uploaded_at", -1)
        .limit(limit)
    )

    return filings


def get_recent_metrics(limit: int = 10) -> List[Dict]:
    """
    Get the most recent metric records.

    Args:
        limit: Maximum number of records to return

    Returns:
        List of metric documents (as dicts)
    """
    db = get_db()
    metrics_collection = db.metrics

    # Get recent metrics, sorted by extraction date
    metrics = list(
        metrics_collection
        .find({}, {"_id": 0})  # Exclude MongoDB's internal _id
        .sort("extracted_at", -1)
        .limit(limit)
    )

    return metrics


def get_metrics_for_filing(filing_id: str) -> List[Dict]:
    """
    Get all metrics for a specific filing.

    Args:
        filing_id: UUID of the filing

    Returns:
        List of metric documents
    """
    db = get_db()
    metrics_collection = db.metrics

    metrics = list(
        metrics_collection
        .find({"filing_id": filing_id}, {"_id": 0})
        .sort("name", 1)
    )

    return metrics


def get_database_stats() -> Dict:
    """
    Get simple statistics about the database.

    Returns:
        Dict with count information
    """
    db = get_db()

    stats = {
        "total_filings": db.filings.count_documents({}),
        "total_metrics": db.metrics.count_documents({})
    }

    return stats
