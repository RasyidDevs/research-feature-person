"""Multi-image uploader + camera capture component with Describe! button."""

from __future__ import annotations

import io
import streamlit as st
from PIL import Image


# ── Thin wrapper so camera captures look like UploadedFile ───────────────

class CameraImage:
    """Minimal file-like wrapper around a camera-captured image."""

    def __init__(self, data: bytes, name: str = "camera_capture.jpg"):
        self._data = data
        self.name = name
        self.size = len(data)

    def read(self, *args, **kwargs):
        return self._data

    def seek(self, *_args, **_kwargs):
        pass

    def getvalue(self):
        return self._data


# ── JS snippet to toggle camera facing mode ─────────────────────────────

_FACING_JS = """
<script>
(function() {
    const FACING = "__FACING__";
    async function switchCamera() {
        const videos = window.parent.document.querySelectorAll("video");
        for (const video of videos) {
            if (video.srcObject) {
                video.srcObject.getTracks().forEach(t => t.stop());
            }
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: { ideal: FACING } }
                });
                video.srcObject = stream;
            } catch(e) { console.warn("Camera switch failed:", e); }
        }
    }
    // small delay to let Streamlit render the video element first
    setTimeout(switchCamera, 800);
})();
</script>
"""


def render_uploader() -> tuple[list | None, bool]:
    """
    Render a tabbed input area (Upload / Camera) and a 'Describe!' button.

    Returns
    -------
    (uploaded_files, should_run)
        uploaded_files : list of UploadedFile / CameraImage or None
        should_run     : True when user clicked "Describe!"
    """
    all_images: list = []

    tab_upload, tab_camera = st.tabs(["📁 Upload File", "📷 Camera"])

    # ── Tab 1: File upload ───────────────────────────────────────────
    with tab_upload:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drag & drop images here",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            help="Upload images",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded:
            all_images.extend(uploaded)

    # ── Tab 2: Camera capture ────────────────────────────────────────
    with tab_camera:
        facing = st.radio(
            "Camera",
            ["🤳 Front Camera", "📸 Back Camera"],
            horizontal=True,
            label_visibility="collapsed",
        )
        facing_mode = "user" if "Front" in facing else "environment"

        # Inject JS to apply facingMode
        st.markdown(
            _FACING_JS.replace("__FACING__", facing_mode),
            unsafe_allow_html=True,
        )

        photo = st.camera_input("Take a photo", key="camera_input")

        if photo is not None:
            raw = photo.getvalue()
            cam_count = st.session_state.get("_cam_seq", 0) + 1
            st.session_state["_cam_seq"] = cam_count
            cam_img = CameraImage(raw, name=f"camera_{cam_count}.jpg")
            all_images.append(cam_img)

    # ── Summary & button ─────────────────────────────────────────────
    if not all_images:
        st.markdown(
            '<p class="empty-hint">Upload or capture one or more images to get started.</p>',
            unsafe_allow_html=True,
        )
        return None, False

    st.markdown(
        f'<p class="file-count">📎 <strong>{len(all_images)}</strong> image(s) ready</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        should_run = st.button("🔍 Describe!", use_container_width=True, type="primary")

    return all_images, should_run
