"""Main Streamlit application for financial document analysis with RAG."""

import streamlit as st
from dotenv import load_dotenv
import os
from typing import List
import numpy as np

from models import Metric, Report, ChatMessage
from ingestion import ingest_pdf
from vlm_client import extract_metrics_with_vlm, merge_duplicate_metrics
from llm_analysis import analyze_metrics_with_gpt, answer_with_rag
from rag_chunking import chunk_text
from rag_retrieval import embed_all_chunks, retrieve_top_k
from database.persistence import (
    create_filing_record,
    save_metrics,
    get_recent_filings,
    get_recent_metrics,
    get_database_stats
)
from database.mongo_client import test_connection

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Financial Document Analyzer",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "report" not in st.session_state:
    st.session_state.report = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "filing_id" not in st.session_state:
    st.session_state.filing_id = None
if "company_info" not in st.session_state:
    st.session_state.company_info = None
if "saved_to_db" not in st.session_state:
    st.session_state.saved_to_db = False


def reset_analysis():
    """Reset all analysis data when a new file is uploaded."""
    st.session_state.metrics = None
    st.session_state.report = None
    st.session_state.chat_history = []
    st.session_state.analysis_complete = False
    st.session_state.chunks = None
    st.session_state.vectors = None
    st.session_state.filing_id = None
    st.session_state.company_info = None
    st.session_state.saved_to_db = False


def main():
    st.title("📊 Financial Document Analyzer")
    st.markdown("Upload a 10-K or financial PDF to extract metrics and get AI-powered insights.")

    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("Document Upload")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a 10-K or other financial document"
        )

        if uploaded_file is not None:
            # Check if it's a new file
            if st.session_state.pdf_name != uploaded_file.name:
                st.session_state.pdf_bytes = uploaded_file.read()
                st.session_state.pdf_name = uploaded_file.name
                reset_analysis()

            st.success(f"Loaded: {uploaded_file.name}")

            # Company information form
            st.subheader("Company Information")
            company_name = st.text_input("Company Name", placeholder="e.g., Apple Inc.")
            ticker = st.text_input("Ticker Symbol", placeholder="e.g., AAPL")
            fiscal_year = st.number_input("Fiscal Year", min_value=2000, max_value=2030, value=2023)

            # Analysis button
            if st.button("🔍 Analyze Document", type="primary", use_container_width=True):
                # Validate company info
                if not company_name or not ticker:
                    st.error("Please provide company name and ticker symbol")
                elif st.session_state.pdf_bytes:
                    # Store company info in session
                    st.session_state.company_info = {
                        "company": company_name,
                        "ticker": ticker,
                        "year": fiscal_year
                    }

                    with st.spinner("Analyzing document..."):
                        try:
                            # Create filing record before analysis
                            filing_id = create_filing_record(
                                company=company_name,
                                ticker=ticker,
                                year=fiscal_year,
                                filename=st.session_state.pdf_name
                            )
                            st.session_state.filing_id = filing_id

                            # Step 1: Ingest PDF (all pages with smart financial table detection)
                            st.info("Step 1/5: Scanning document and filtering for financial tables...")
                            pages, stats = ingest_pdf(st.session_state.pdf_bytes, max_pages=None, use_llm_filter=True)

                            # Build success message
                            success_msg = f"✓ Scanned {stats['processed_pages']} pages\n\n"
                            success_msg += f"  • {stats['pages_with_tables']} pages with financial tables\n\n"

                            if stats.get('pages_filtered_out', 0) > 0:
                                success_msg += f"  • {stats['pages_filtered_out']} non-financial pages filtered out\n\n"

                            success_msg += f"  • {stats['pages_without_tables']} text-only pages"

                            st.success(success_msg)

                            # Step 2: Extract metrics with GPT-4o-mini vision (only from table pages)
                            if stats['pages_with_tables'] > 0:
                                st.info(f"Step 2/3: Extracting metrics from {stats['pages_with_tables']} pages with GPT-4o-mini vision...")
                                metrics, errors = extract_metrics_with_vlm(pages)
                                metrics = merge_duplicate_metrics(metrics)
                                st.session_state.metrics = metrics

                                # Show any errors
                                if errors:
                                    with st.expander("⚠️ VLM Processing Warnings", expanded=False):
                                        for error in errors:
                                            st.warning(error)

                                st.success(f"✓ Extracted {len(metrics)} unique metrics")
                            else:
                                st.warning("No pages with tables found. Skipping VLM extraction.")
                                st.session_state.metrics = []

                            # Step 3: Create text chunks for RAG
                            st.info("Step 3/5: Creating text chunks for semantic search...")
                            chunks = chunk_text(pages, chunk_size=500, overlap=50)
                            st.session_state.chunks = chunks
                            st.success(f"✓ Created {len(chunks)} text chunks")

                            # Step 4: Generate embeddings
                            if chunks:
                                st.info("Step 4/5: Generating embeddings for chunks...")
                                vectors = embed_all_chunks(chunks)
                                st.session_state.vectors = vectors
                                st.success(f"✓ Embedded {len(chunks)} chunks")
                            else:
                                st.session_state.vectors = np.array([])
                                st.warning("No chunks to embed")

                            # Step 5: Analyze with GPT
                            if st.session_state.metrics:
                                st.info("Step 5/5: Analyzing metrics with GPT...")
                                report = analyze_metrics_with_gpt(st.session_state.metrics)
                                st.session_state.report = report
                                st.success("✓ Analysis complete!")
                            else:
                                # Create empty report if no metrics
                                from models import Report
                                st.session_state.report = Report(
                                    summary="No financial metrics were extracted from this document.",
                                    key_signals=[],
                                    risk_flags=[]
                                )
                                st.info("No metrics found to analyze.")

                            st.session_state.analysis_complete = True

                        except Exception as e:
                            st.error(f"Error during analysis: {str(e)}")
                            st.exception(e)
                else:
                    st.warning("Please upload a PDF file first")

        st.divider()

        # Settings
        st.header("Settings")
        st.caption("API keys are loaded from .env file")

        # Check if API key is set
        openai_key = os.getenv("OPENAI_API_KEY")

        if openai_key:
            st.success("✓ OpenAI API key loaded")
            st.caption("Used for: Vision, Analysis, Chat")
        else:
            st.error("✗ OpenAI API key missing")
            st.caption("Add OPENAI_API_KEY to .env file")

    # Main content area
    if not st.session_state.pdf_bytes:
        # Welcome screen
        st.info("👈 Upload a PDF file from the sidebar to get started")

        st.markdown("""
        ### How it works:

        1. **Upload** a financial PDF (10-K, annual report, etc.)
        2. **Smart Filtering** - LLM identifies only financial tables
        3. **Vision Extraction** - GPT-4o-mini reads tables and charts
        4. **Text Chunking** - Creates semantic chunks for document search
        5. **Embedding** - Generates vector embeddings for RAG
        6. **Analysis** - GPT generates insights and risk flags
        7. **RAG Chat** - Ask questions about the entire document

        ### Features:

        - 🎯 Smart financial table detection (filters out TOC, exhibits, etc.)
        - 👁️ GPT-4o-mini vision extracts metrics from complex tables
        - 🔍 **RAG-powered chat** - Search entire document, not just metrics
        - 📊 Automatic financial analysis with key signals and risk flags
        - 📎 Source citations showing which pages answers came from
        - 💬 Interactive chat to explore both structured and unstructured data
        - 📄 Processes entire PDF (not just first few pages)
        """)

    elif st.session_state.analysis_complete and st.session_state.metrics and st.session_state.report:
        # Show analysis results
        st.header(f"Analysis: {st.session_state.pdf_name}")

        # Metrics section
        st.subheader("📈 Extracted Metrics")

        if st.session_state.metrics:
            # Create a formatted display for metrics
            for metric in st.session_state.metrics:
                with st.expander(f"**{metric.name}**", expanded=True):
                    # Display as a simple table
                    cols = st.columns(len(metric.values))
                    for i, value_data in enumerate(metric.values):
                        with cols[i]:
                            st.metric(
                                label=value_data["period"],
                                value=f"{value_data['value']:,.0f}" if isinstance(value_data['value'], (int, float)) else value_data['value']
                            )
        else:
            st.info("No metrics extracted")

        # Save to Database button
        if st.session_state.filing_id and st.session_state.metrics and not st.session_state.saved_to_db:
            if st.button("💾 Save to Database", type="secondary"):
                try:
                    with st.spinner("Saving to MongoDB..."):
                        num_saved = save_metrics(
                            filing_id=st.session_state.filing_id,
                            metrics=st.session_state.metrics
                        )
                        st.session_state.saved_to_db = True
                        st.success(f"✓ Saved {num_saved} metric records to database!")
                        st.info(f"Filing ID: `{st.session_state.filing_id}`")
                except Exception as e:
                    st.error(f"Error saving to database: {str(e)}")
        elif st.session_state.saved_to_db:
            st.success("✓ Already saved to database")
            st.info(f"Filing ID: `{st.session_state.filing_id}`")

        st.divider()

        # Analysis section
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Key Signals")
            report = st.session_state.report
            if report.key_signals:
                for signal in report.key_signals:
                    st.markdown(f"- {signal}")
            else:
                st.info("No key signals identified")

        with col2:
            st.subheader("⚠️ Risk Flags")
            if report.risk_flags:
                for flag in report.risk_flags:
                    st.markdown(f"- {flag}")
            else:
                st.info("No risk flags identified")

        st.divider()

        # Summary
        st.subheader("📝 Summary")
        st.write(report.summary)

        st.divider()

        # Preview Database Records section
        st.subheader("🗄️ Database Preview")

        try:
            # Test MongoDB connection
            if test_connection():
                # Get database stats
                stats = get_database_stats()
                st.info(f"📊 Database contains {stats['total_filings']} filings and {stats['total_metrics']} metrics")

                # Create tabs for filings and metrics
                tab1, tab2 = st.tabs(["Recent Filings", "Recent Metrics"])

                with tab1:
                    filings = get_recent_filings(limit=5)
                    if filings:
                        for filing in filings:
                            with st.expander(f"{filing['ticker']} - {filing['year']} ({filing['filename']})", expanded=False):
                                st.json(filing)
                    else:
                        st.info("No filings in database yet")

                with tab2:
                    metrics = get_recent_metrics(limit=10)
                    if metrics:
                        for metric in metrics:
                            st.json(metric)
                    else:
                        st.info("No metrics in database yet")
            else:
                st.warning("⚠️ MongoDB not connected. Database features unavailable.")
                st.caption("Make sure MongoDB is running and MONGODB_URI is set in .env")
        except Exception as e:
            st.warning(f"Database preview unavailable: {str(e)}")

        st.divider()

        # Chat interface
        st.subheader("💬 Ask Questions")

        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message.role == "user":
                    st.chat_message("user").write(message.content)
                else:
                    st.chat_message("assistant").write(message.content)

        # Chat input
        user_question = st.chat_input("Ask a question about this document...")

        if user_question:
            # Add user message to history
            user_msg = ChatMessage(role="user", content=user_question)
            st.session_state.chat_history.append(user_msg)

            # Display user message
            st.chat_message("user").write(user_question)

            # Get assistant response with RAG
            with st.spinner("Searching document and thinking..."):
                try:
                    # Retrieve relevant chunks
                    retrieved_chunks = []
                    if st.session_state.chunks and st.session_state.vectors is not None and len(st.session_state.vectors) > 0:
                        retrieved_chunks = retrieve_top_k(
                            query=user_question,
                            chunks=st.session_state.chunks,
                            vectors=st.session_state.vectors,
                            k=2
                        )

                    # Get answer with RAG
                    response = answer_with_rag(
                        question=user_question,
                        metrics=st.session_state.metrics or [],
                        report=st.session_state.report,
                        retrieved_chunks=retrieved_chunks,
                        chat_history=st.session_state.chat_history
                    )

                    # Add assistant message to history
                    assistant_msg = ChatMessage(role="assistant", content=response["answer"])
                    st.session_state.chat_history.append(assistant_msg)

                    # Display assistant message
                    with st.chat_message("assistant"):
                        st.write(response["answer"])

                        # Show sources if available
                        if response["sources"]:
                            st.caption(f"📎 Sources: Pages {', '.join(map(str, response['sources']))}")

                except Exception as e:
                    st.error(f"Error getting response: {str(e)}")
                    st.exception(e)

    else:
        # File uploaded but not analyzed yet
        st.info("👆 Click 'Analyze Document' in the sidebar to start analysis")


if __name__ == "__main__":
    main()
