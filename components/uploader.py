"""Multi-image uploader + camera capture component."""

from __future__ import annotations

import io
import streamlit as st
from PIL import Image
from typing import List, Tuple


# ── Thin wrapper so camera captures look like UploadedFile ───

class CameraImage:
    """Minimal file-like wrapper around a camera-captured image."""

    def __init__(self, data: bytes, name: str = "camera_capture.jpg"):
        self._data = data
        self.name = name
        self.size = len(data)

    def read(self) -> bytes:
        return self._data

    def getvalue(self) -> bytes:
        return self._data

    def seek(self, _: int) -> None:
        pass


def _open_image(f) -> Image.Image:
    """Open an image from an UploadedFile or CameraImage."""
    if hasattr(f, "seek"):
        f.seek(0)
    raw = f.read() if callable(getattr(f, "read", None)) else f.getvalue()
    return Image.open(io.BytesIO(raw)).convert("RGB")


def render_uploader() -> Tuple[List[Image.Image], List[str]]:
    """
    Render image upload + camera capture UI.
    Returns (list_of_pil_images, list_of_filenames).
    """
    images: List[Image.Image] = []
    filenames: List[str] = []

    tab_upload, tab_camera = st.tabs(["📁 Upload Files", "📷 Camera"])

    with tab_upload:
        uploaded = st.file_uploader(
            "Select images",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded:
            for f in uploaded:
                try:
                    img = _open_image(f)
                    images.append(img)
                    filenames.append(f.name)
                except Exception:
                    st.warning(f"⚠️ Could not open: {f.name}")

    with tab_camera:
        cam_col1, cam_col2 = st.columns(2)
        with cam_col1:
            camera_img = st.camera_input("Take a photo")
        with cam_col2:
            if camera_img:
                st.success("✅ Photo captured!")

        if camera_img:
            try:
                cam_wrapper = CameraImage(camera_img.getvalue(), "camera_capture.jpg")
                img = _open_image(cam_wrapper)
                images.append(img)
                filenames.append("camera_capture.jpg")
            except Exception:
                st.warning("⚠️ Could not process camera image")

    return images, filenames
