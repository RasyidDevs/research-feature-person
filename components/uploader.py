"""Multi-image uploader component with Describe! button."""

from __future__ import annotations
import streamlit as st


def render_uploader() -> tuple[list | None, bool]:
    """
    Render the upload area and a 'Describe!' button.

    Returns
    -------
    (uploaded_files, should_run)
        uploaded_files : list of UploadedFile or None
        should_run     : True when user clicked "Describe!"
    """
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drag & drop images here",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="Upload images",
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if not uploaded:
        st.markdown(
            '<p class="empty-hint">Upload one or more images to get started.</p>',
            unsafe_allow_html=True,
        )
        return None, False

    st.markdown(
        f'<p class="file-count">📎 <strong>{len(uploaded)}</strong> image(s) uploaded</p>',
        unsafe_allow_html=True,
    )

    # ── Describe! button ─────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        should_run = st.button("🔍 Describe!", use_container_width=True, type="primary")

    return uploaded, should_run
