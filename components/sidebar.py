"""Sidebar — app info & mode guide."""

import streamlit as st


def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ About")
        st.markdown("---")

        st.markdown(
            """
            **Human Feature Analyzer** uses AI to detect and count
            features/accessories on people in your images.

            ### How it works
            1. 📷 Upload images with people
            2. 🎯 Select analysis mode
            3. ✏️ Type your question
            4. 🚀 Click Analyze
            5. 📊 View results & download CSV
            """
        )

        st.markdown("---")
        st.markdown("### 🎯 Modes Explained")
        st.markdown(
            """
            | Mode | What it sends to AI |
            |------|---------------------|
            | **Fullbody** | Person body crop only |
            | **Head and Fullbody** | Both face + body crops |
            | **Head** | Face/head crop only |
            """
        )

        st.markdown("---")
        st.markdown("### 📊 Output Values")
        st.markdown(
            """
            | Value | Meaning |
            |-------|---------|
            | **1** | Present / True |
            | **0** | Not present / False |
            | **-1** | Unknown / Ambiguous |
            """
        )

        st.markdown("---")
        st.caption("Powered by YOLOv8 + GPT-4.1-mini")
