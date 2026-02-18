"""
Human Descriptor System — Streamlit UI
=======================================
Upload images → Select mode → Input question → Analyze → Results + CSV
"""

import io
import streamlit as st
from components.page_config import setup_page
from components.sidebar import render_sidebar
from components.uploader import render_uploader
from components.results import render_results
from styles.theme import inject_css
from ultralytics import YOLO
from langchain_openai import ChatOpenAI


# ── Model loading (cached) ──────────────────────────────────

@st.cache_resource
def load_seg_model():
    """Load YOLOv8m-seg for person segmentation."""
    return YOLO("models/yolov8m-seg.pt")


@st.cache_resource
def load_head_model():
    """Load YOLO head/face detection model."""
    return YOLO("models/yolo-face-person.pt")


@st.cache_resource
def load_llm():
    """Load LangChain ChatOpenAI."""
    return ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        api_key=st.secrets["OPENAI_API_KEY"],
    )


# ── Main ─────────────────────────────────────────────────────

def main():
    setup_page()
    inject_css()
    # Header
    st.markdown(
        """
        <div style="text-align:center; padding: 1.5rem 0 0.5rem;">
            <h1 style="margin:0;">🔍 Human Feature Analyzer</h1>
            <p style="opacity:0.7; margin-top:0.3rem;">
                Upload images · Select mode · Ask questions · Get structured results
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    render_sidebar()

    # Load models
    seg_model = load_seg_model()
    head_model = load_head_model()
    llm = load_llm()

    # ── Step 1: Upload images ────────────────────────────────
    st.markdown("### 📷 Step 1 — Upload Images")
    images, filenames = render_uploader()

    if not images:
        st.info("👆 Upload one or more images to begin.", icon="📸")
        return

    # Show preview
    cols = st.columns(min(len(images), 4))
    for idx, (img, fname) in enumerate(zip(images, filenames)):
        with cols[idx % len(cols)]:
            st.image(img, caption=fname, use_container_width=True)

    # ── Step 2: Mode selection ───────────────────────────────
    st.markdown("### 🎯 Step 2 — Select Mode")
    mode = st.selectbox(
        "Analysis mode",
        options=["Head and Fullbody","Fullbody",  "Head"],
        help=(
            "**Fullbody** — Analyze person body only\n\n"
            "**Head and Fullbody** — Analyze both head/face and body\n\n"
            "**Head** — Analyze head/face only"
        ),
    )

    # ── Step 3: Input question ───────────────────────────────
    st.markdown("### ✏️ Step 3 — Input Question")
    question = st.text_area(
        "What do you want to detect?",
        placeholder="e.g. Is this person wearing glasses? Is this person fat or thin?",
        height=80,
    )

    # ── Step 4: Analyze ──────────────────────────────────────
    st.markdown("---")
    analyze_btn = st.button(
        "🚀 Analyze All Images",
        type="primary",
        use_container_width=True,
        disabled=not question.strip(),
    )

    if analyze_btn and question.strip():
        from inference.descriptor import run_pipeline

        with st.spinner("🔄 Running analysis pipeline..."):
            results = run_pipeline(
                images=images,
                filenames=filenames,
                mode=mode,
                question=question.strip(),
                seg_model=seg_model,
                head_model=head_model,
                llm=llm,
            )
            
        st.session_state["analysis_results"] = results

    # ── Step 5: Show results ─────────────────────────────────
    if "analysis_results" in st.session_state:
        render_results(st.session_state["analysis_results"])


if __name__ == "__main__":
    main()
