"""MongoDB connection helper."""

import os
from pymongo import MongoClient
from pymongo.database import Database


def get_db() -> Database:
    """
    Get a connected MongoDB database instance.

    Uses environment variables:
    - MONGODB_URI: MongoDB connection string (default: mongodb://localhost:27017/)
    - MONGODB_DB_NAME: Database name (default: financial_analyzer)

    Returns:
        MongoDB database instance
    """
    # Get connection settings from environment
    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    db_name = os.getenv("MONGODB_DB_NAME", "financial_analyzer")

    # Create client with SSL settings for MongoDB Atlas
    # For production, you should enable SSL verification
    client = MongoClient(
        mongodb_uri,
        tlsAllowInvalidCertificates=True  # For development only
    )
    db = client[db_name]

    return db


def test_connection() -> bool:
    """
    Test if MongoDB connection is working.

    Returns:
        True if connection successful, False otherwise
    """
    try:
        db = get_db()
        # Ping the database
        db.command('ping')
        return True
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        return False
