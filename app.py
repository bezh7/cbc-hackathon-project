"""Main Streamlit application for financial document analysis."""

import streamlit as st
from dotenv import load_dotenv
import os
from typing import List

from models import Metric, Report, ChatMessage
from ingestion import ingest_pdf
from vlm_client import extract_metrics_with_vlm, merge_duplicate_metrics
from llm_client import analyze_metrics_with_gpt, answer_chat_question

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


def reset_analysis():
    """Reset all analysis data when a new file is uploaded."""
    st.session_state.metrics = None
    st.session_state.report = None
    st.session_state.chat_history = []
    st.session_state.analysis_complete = False


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

            # Analysis button
            if st.button("🔍 Analyze Document", type="primary", use_container_width=True):
                if st.session_state.pdf_bytes:
                    with st.spinner("Analyzing document..."):
                        try:
                            # Step 1: Ingest PDF
                            st.info("Step 1/3: Extracting pages from PDF...")
                            pages = ingest_pdf(st.session_state.pdf_bytes, max_pages=3)
                            st.success(f"Extracted {len(pages)} pages")

                            # Step 2: Extract metrics with VLM
                            st.info("Step 2/3: Extracting metrics with VLM...")
                            metrics = extract_metrics_with_vlm(pages)
                            metrics = merge_duplicate_metrics(metrics)
                            st.session_state.metrics = metrics
                            st.success(f"Extracted {len(metrics)} unique metrics")

                            # Step 3: Analyze with GPT
                            st.info("Step 3/3: Analyzing metrics with GPT...")
                            report = analyze_metrics_with_gpt(metrics)
                            st.session_state.report = report
                            st.success("Analysis complete!")

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

        # Check if API keys are set
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if deepseek_key:
            st.success("✓ DeepSeek API key loaded")
        else:
            st.error("✗ DeepSeek API key missing")

        if openai_key:
            st.success("✓ OpenAI API key loaded")
        else:
            st.error("✗ OpenAI API key missing")

    # Main content area
    if not st.session_state.pdf_bytes:
        # Welcome screen
        st.info("👈 Upload a PDF file from the sidebar to get started")

        st.markdown("""
        ### How it works:

        1. **Upload** a financial PDF (10-K, annual report, etc.)
        2. **Analyze** the document with AI to extract metrics
        3. **Review** key financial signals and risk flags
        4. **Chat** to ask questions about the document

        ### Features:

        - 🔍 Vision AI extracts metrics from tables and charts
        - 📊 Automatic financial analysis with GPT
        - 💬 Interactive chat to explore the data
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

            # Get assistant response
            with st.spinner("Thinking..."):
                try:
                    answer = answer_chat_question(
                        question=user_question,
                        metrics=st.session_state.metrics,
                        report=st.session_state.report,
                        chat_history=st.session_state.chat_history
                    )

                    # Add assistant message to history
                    assistant_msg = ChatMessage(role="assistant", content=answer)
                    st.session_state.chat_history.append(assistant_msg)

                    # Display assistant message
                    st.chat_message("assistant").write(answer)

                except Exception as e:
                    st.error(f"Error getting response: {str(e)}")

    else:
        # File uploaded but not analyzed yet
        st.info("👆 Click 'Analyze Document' in the sidebar to start analysis")


if __name__ == "__main__":
    main()
