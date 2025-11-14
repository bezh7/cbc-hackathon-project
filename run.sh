#!/bin/bash

# Financial Document Analyzer - Run Script

echo "🚀 Starting Financial Document Analyzer..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found!"
    echo "📝 Please copy .env.example to .env and add your API keys"
    echo ""
    echo "cp .env.example .env"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📥 Checking dependencies..."
pip install -q -r requirements.txt

# Run Streamlit app
echo "✓ Starting Streamlit app..."
echo "📊 Open your browser to: http://localhost:8501"
echo ""

streamlit run frontend/app.py
