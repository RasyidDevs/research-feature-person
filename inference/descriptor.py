"""
Descriptor Inference
====================
YOLO detection + OpenAI vision descriptor.

Functions:
- run_yolo_only:       YOLO detection → bounding box results (fast, no API call)
- draw_bounding_boxes: overlay colored boxes + labels on a PIL image copy
- run_descriptor:      full pipeline (YOLO + crop + OpenAI)
"""
import json
import io
import os
import base64

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Union, List, Any, Dict, Optional
from openai import OpenAI

from utils.prompt import PROMPT_TEMPLATE


# ── Colour palette for bounding boxes ────────────────────────────────────
_BOX_COLORS = {
    "head":   (88, 166, 255),   # blue
    "person": (63, 185, 80),    # green
}
_DEFAULT_COLOR = (167, 139, 250)  # purple fallback


# ── Image conversion helpers ─────────────────────────────────────────────

def pil_to_data_url(img_pil: Image.Image, fmt: str = "JPEG") -> str:
    """Convert PIL Image → data:image/…;base64,… URL."""
    buf = io.BytesIO()
    img_pil.convert("RGB").save(buf, format=fmt.upper())
    b64 = base64.b64encode(buf.getvalue()).decode()
    mime = "jpeg" if fmt.upper() == "JPEG" else "png"
    return f"data:image/{mime};base64,{b64}"


def _to_bgr(img_any) -> np.ndarray:
    if isinstance(img_any, np.ndarray):
        arr = img_any
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        return arr
    if isinstance(img_any, Image.Image):
        return cv2.cvtColor(np.array(img_any.convert("RGB")), cv2.COLOR_RGB2BGR)
    raise TypeError("Unsupported image type")


def _area_xyxy(xyxy):
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _normalize_inputs(inputs):
    if isinstance(inputs, list):
        return inputs
    if isinstance(inputs, str):
        if os.path.isfile(inputs) and inputs.lower().endswith(".txt"):
            with open(inputs, "r", encoding="utf-8") as f:
                return [ln.strip() for ln in f if ln.strip()]
        return [inputs]
    return [inputs]


# ── YOLO-only detection (no OpenAI) ──────────────────────────────────────

def run_yolo_only(
    image: Image.Image,
    model,
    conf: float = 0.5,
    imgsz: int = 640,
) -> Dict[str, Any]:
    """
    Run YOLO on a single PIL image.
    Returns dict with 'per_class' mapping class_id → {class_name, conf, xyxy, area}.
    Keeps only the biggest bbox per class.
    """
    bgr = _to_bgr(image)
    results = model.predict(
        source=[bgr], conf=conf, iou=0.7,
        imgsz=imgsz, verbose=False,
    )
    names = getattr(model, "names", None)
    best: Dict[int, Dict[str, Any]] = {}

    r = results[0]
    if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
        xyxys = r.boxes.xyxy.cpu().tolist()
        confs = r.boxes.conf.cpu().tolist()
        clss  = r.boxes.cls.cpu().tolist()

        for xyxy, cf, c in zip(xyxys, confs, clss):
            cid = int(c)
            area = _area_xyxy(xyxy)
            if (cid not in best) or (area > best[cid]["area"]):
                if isinstance(names, dict):
                    cname = names.get(cid, str(cid))
                elif isinstance(names, (list, tuple)) and cid < len(names):
                    cname = names[cid]
                else:
                    cname = str(cid)
                best[cid] = {
                    "class_name": cname,
                    "conf": float(cf),
                    "xyxy": [float(v) for v in xyxy],
                    "area": float(area),
                }

    return {"per_class": best}


# ── Draw bounding boxes on image ─────────────────────────────────────────

def draw_bounding_boxes(image: Image.Image, yolo_result: Dict) -> Image.Image:
    """
    Draw YOLO bounding boxes + labels on a *copy* of the image.
    Returns a new PIL Image with overlays.
    """
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    # Try to load a nicer font, fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, img.width // 40))
    except (OSError, IOError):
        font = ImageFont.load_default()

    per_class = yolo_result.get("per_class", {})
    for _cid, det in per_class.items():
        x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
        cname = det.get("class_name", "?")
        conf  = det.get("conf", 0)
        color = _BOX_COLORS.get(cname.lower(), _DEFAULT_COLOR)

        # Box outline (3px)
        for i in range(3):
            draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color)

        # Label background
        label = f"{cname} {conf:.0%}"
        bbox = draw.textbbox((x1, y1), label, font=font)
        lw, lh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([x1, y1 - lh - 8, x1 + lw + 8, y1], fill=color)
        draw.text((x1 + 4, y1 - lh - 6), label, fill=(255, 255, 255), font=font)

    return img


# ── Crop helpers ─────────────────────────────────────────────────────────

def _crop_face_person(image: Image.Image, yolo_result: Dict):
    """Crop face and person regions from a single YOLO result."""
    per_class = yolo_result.get("per_class", {})
    w, h = image.size
    img_face = None
    img_person = None

    for _cid, det in per_class.items():
        x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
        x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
        x2, y2 = max(0, min(x2, w)),     max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = image.crop((x1, y1, x2, y2))
        cname = det.get("class_name", "").lower()
        if cname == "head":
            img_face = crop
        elif cname == "person":
            img_person = crop

    return img_face, img_person


# ── OpenAI vision call ───────────────────────────────────────────────────

def _hit_openai(
    image_face: Optional[Image.Image],
    image_person: Optional[Image.Image],
    client: OpenAI,
) -> dict:
    """Send face + person crops to GPT-4.1-mini for attribute description."""
    content = [{"type": "input_text", "text": PROMPT_TEMPLATE}]

    if image_face is not None:
        content.append({
            "type": "input_image",
            "image_url": pil_to_data_url(image_face, "jpeg"),
        })
    if image_person is not None:
        content.append({
            "type": "input_image",
            "image_url": pil_to_data_url(image_person, "jpeg"),
        })

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[{"role": "user", "content": content}],
        max_output_tokens=64,
    )

    raw = response.output_text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"Face": [], "Body": raw}


# ── Full pipeline ────────────────────────────────────────────────────────

def run_descriptor(
    image: Image.Image,
    filename: str,
    model,
    client: OpenAI,
    yolo_result: Optional[Dict] = None,
) -> dict:
    """
    Full inference: YOLO detect → crop → OpenAI describe.
    If yolo_result is provided, skips re-running YOLO.
    """
    if yolo_result is None:
        yolo_result = run_yolo_only(image, model)

    img_face, img_person = _crop_face_person(image, yolo_result)
    return _hit_openai(img_face, img_person, client)
