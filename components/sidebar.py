"""Sidebar — app info & settings."""

import streamlit as st


def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        st.markdown("---")

        st.markdown("### Image Types")
        st.markdown(
            """
            | Type | Descriptor Output |
            |------|-------------------|
            | **Head** | Face accessories list |
            | **Person** | Body type classification |
            """
        )

        st.markdown("---")
        st.markdown("### Body Classifications")
        st.markdown("`skinny`  ·  `average`  ·  `fat`")

        st.markdown("---")
        st.markdown(
            '<p style="opacity:0.5; font-size:0.78rem;">Human Descriptor v1.0</p>',
            unsafe_allow_html=True,
        )
