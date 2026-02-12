"""
Human Descriptor System — Streamlit UI
=======================================
Upload head / full-body images and receive attribute descriptors.
"""

import streamlit as st
from components.page_config import setup_page
from components.sidebar import render_sidebar
from components.uploader import render_uploader
from components.results import render_preview, render_results
from styles.theme import inject_css
from ultralytics import YOLO
from openai import OpenAI
import os



@st.cache_resource
def load_model():
    return YOLO("models/yolo-face-person.pt")


@st.cache_resource
def load_client():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def main():
    model = load_model()
    client = load_client()
    setup_page()
    inject_css()

    # ── Header ──────────────────────────────────────────────────────
    st.markdown(
        '<h1 class="main-title">Human Descriptor</h1>'
        '<p class="subtitle">Upload images or use your camera to extract face accessories & body type descriptors</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Sidebar ─────────────────────────────────────────────────────
    render_sidebar()

    # ── Upload section ──────────────────────────────────────────────
    uploaded_files, should_run = render_uploader()

    # ── Track inference state ───────────────────────────────────────
    if "inference_done" not in st.session_state:
        st.session_state["inference_done"] = False

    # Reset caches when files change
    current_keys = set()
    if uploaded_files:
        current_keys = {f"{getattr(f, 'name', 'cam')}_{getattr(f, 'size', 0)}" for f in uploaded_files}

    prev_keys = st.session_state.get("prev_file_keys", set())
    if current_keys != prev_keys:
        st.session_state["prev_file_keys"] = current_keys
        st.session_state["inference_done"] = False
        st.session_state.pop("inference_results", None)
        st.session_state.pop("yolo_cache", None)

    # ── Preview / Results ───────────────────────────────────────────
    if uploaded_files:
        if should_run:
            st.session_state["inference_done"] = True

        if st.session_state["inference_done"]:
            render_results(uploaded_files, model, client)
        else:
            render_preview(uploaded_files, model)


if __name__ == "__main__":
    main()
