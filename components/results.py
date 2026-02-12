"""Results display — YOLO preview cards + descriptor output."""

from __future__ import annotations

import base64
import io
import streamlit as st
from PIL import Image

from inference.descriptor import run_yolo_only, draw_bounding_boxes, run_descriptor


# ── Helpers ─────────────────────────────────────────────────────────────

def _pil_to_b64_img_tag(img: Image.Image, alt: str = "") -> str:
    """Convert PIL image to an <img> tag with inline base64 src."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/jpeg;base64,{b64}" alt="{alt}" class="card-img"/>'


def _render_face_result(accessories: list[str]):
    if not accessories:
        return '<div class="tag-empty">No accessories detected</div>'
    tags = "".join(f'<span class="tag">{acc}</span>' for acc in accessories)
    return f'<div class="tag-row">{tags}</div>'


def _body_class_color(body_type: str) -> str:
    mapping = {"skinny": "#3b82f6", "thin": "#3b82f6", "average": "#22c55e", "fat": "#f97316"}
    return mapping.get(body_type.lower(), "#94a3b8")


def _render_body_result(body_type: str) -> str:
    color = _body_class_color(body_type)
    return (
        f'<div class="body-badge" style="--badge-color:{color};">'
        f"{body_type.capitalize()}</div>"
    )


# ── Preview grid (YOLO boxes only, no OpenAI) ──────────────────────────

def render_preview(uploaded_files: list, model):
    """
    Show uploaded images with YOLO bounding-box overlays.
    Caches YOLO results in session_state to avoid re-running on rerender.
    """
    st.markdown("---")
    st.markdown('<h2 class="section-title">Preview</h2>', unsafe_allow_html=True)

    # Run YOLO only once per file set
    if "yolo_cache" not in st.session_state:
        st.session_state["yolo_cache"] = {}

    yolo_cache = st.session_state["yolo_cache"]
    cards_html = '<div class="card-grid">'

    for f in uploaded_files:
        image = Image.open(f)

        # Cache key = filename + size (simple dedup)
        cache_key = f"{f.name}_{f.size}"
        if cache_key not in yolo_cache:
            yolo_cache[cache_key] = run_yolo_only(image, model)

        yolo_res = yolo_cache[cache_key]
        annotated = draw_bounding_boxes(image, yolo_res)

        img_tag = _pil_to_b64_img_tag(annotated, alt=f.name)
        cards_html += f"""
        <div class="card">
            {img_tag}
            <p class="card-filename">{f.name}</p>
        </div>"""

    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)


# ── Full results (after Describe! clicked) ──────────────────────────────

def render_results(uploaded_files: list, model, client):
    """Run full inference and display result cards with descriptors."""

    st.markdown("---")
    st.markdown('<h2 class="section-title">Results</h2>', unsafe_allow_html=True)

    yolo_cache = st.session_state.get("yolo_cache", {})

    # Run inference for all files if not already cached
    if "inference_results" not in st.session_state:
        st.session_state["inference_results"] = {}

    inf_cache = st.session_state["inference_results"]
    total = len(uploaded_files)

    # Progress bar
    progress = st.progress(0, text="Running inference…")

    for idx, f in enumerate(uploaded_files):
        cache_key = f"{f.name}_{f.size}"

        if cache_key not in inf_cache:
            image = Image.open(f)
            yolo_res = yolo_cache.get(cache_key)
            result = run_descriptor(image, f.name, model, client, yolo_result=yolo_res)
            inf_cache[cache_key] = {
                "result": result,
                "cache_key": cache_key,
            }

        progress.progress((idx + 1) / total, text=f"Processing {idx + 1}/{total}…")

    progress.empty()

    # Build cards HTML
    cards_html = '<div class="card-grid">'

    for f in uploaded_files:
        image = Image.open(f)
        cache_key = f"{f.name}_{f.size}"

        yolo_res = yolo_cache.get(cache_key)
        if yolo_res:
            annotated = draw_bounding_boxes(image, yolo_res)
        else:
            annotated = image

        img_tag = _pil_to_b64_img_tag(annotated, alt=f.name)
        inf = inf_cache.get(cache_key, {})
        result = inf.get("result", {"Face": [], "Body": "unknown"})

        face_html = _render_face_result(result.get("Face", []))
        body_html = _render_body_result(result.get("Body", "unknown"))

        cards_html += f"""
        <div class="card">
            {img_tag}
            <p class="card-filename">{f.name}</p>
            <p class="card-label">🧑 Face Accessories</p>
            {face_html}
            <p class="card-label">🧍 Body Type</p>
            {body_html}
        </div>"""

    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)
