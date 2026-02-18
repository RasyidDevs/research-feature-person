"""
Descriptor Inference — Full Pipeline
=====================================
Pipeline from exp.ipynb:
1. YOLOv8m-seg  → segment persons from image
2. Head crop    → DeepFace or YOLO head detection
3. LangChain    → GPT-4.1-mini structured output for feature detection

Functions:
- extract_person_crops:  segment persons via YOLOv8m-seg
- crops_yolo_head:       crop heads via YOLO head model
- crops_deepface_head:   crop heads via DeepFace
- pil_to_data_url:       convert PIL Image → base64 data URL
- attribute_calls:       call LLM for feature detection
- run_pipeline:          orchestrate full pipeline per image
"""

import io
import base64
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Literal, Union, Annotated
from PIL import Image
from pydantic import BaseModel, Field, model_validator
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage


# ── Pydantic models for structured LLM output (per-image + per-person) ──
class PersonCounts(BaseModel):
    person_id: str
    counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Mapping feature -> 1/0/-1 for this person",
    )


class ImageAccessoryCounts(BaseModel):
    status: Literal[200, 404]
    per_person: List[PersonCounts] = Field(default_factory=list)
    counts: Dict[str, int] = Field(default_factory=dict)
    reason: Optional[str] = None

    @model_validator(mode="after")
    def enforce_404_reason(self):
        if self.status == 404:
            if not self.reason:
                raise ValueError("reason must be provided when status=404")
            # bikin output 404 clean
            self.per_person = []
            self.counts = {}
        else:
            # optional: kalau 200, reason harus None
            self.reason = None
        return self
# ── Image utilities ──────────────────────────────────────────

def pil_to_data_url(img_pil: Image.Image, fmt: str = "JPEG") -> str:
    """Convert PIL Image → data:image/…;base64,… URL."""
    buf = io.BytesIO()
    img_pil.convert("RGB").save(buf, format=fmt.upper())
    b64 = base64.b64encode(buf.getvalue()).decode()
    mime = "jpeg" if fmt.upper() == "JPEG" else "png"
    return f"data:image/{mime};base64,{b64}"


def _pil_to_cv2(img_pil: Image.Image) -> np.ndarray:
    """Convert PIL Image → OpenCV BGR ndarray."""
    arr = np.array(img_pil.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _ensure_uint8(arr: np.ndarray) -> np.ndarray:
    """Ensure array is uint8 for display/processing."""
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    return arr


# ── Person segmentation ─────────────────────────────────────

def extract_person_crops(
    image: Image.Image,
    seg_model,
    conf: float = 0.6,
    iou: float = 0.7,
) -> List[Image.Image]:
    """
    Segment persons from image using YOLOv8m-seg.
    Returns list of PIL Image crops (one per person, background masked).
    """
    img_cv = _pil_to_cv2(image)
    H, W = img_cv.shape[:2]

    res = seg_model.predict(img_cv, conf=conf, iou=iou, classes=[0], verbose=False)[0]

    if res.masks is None or res.masks.data is None:
        return []

    masks = res.masks.data.cpu().numpy()
    images: List[Image.Image] = []

    for i, box in enumerate(res.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        mask = cv2.resize(masks[i], (W, H))
        mask = (mask > 0.5).astype(np.uint8) * 255

        masked = cv2.bitwise_and(img_cv, img_cv, mask=mask)
        crop = masked[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop = _ensure_uint8(crop)
        images.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))

    return images


# ── Head cropping ────────────────────────────────────────────

def crops_yolo_head(
    person_crops: List[Image.Image],
    model_head,
    conf: float = 0.7,
    iou: float = 0.7,
) -> List[Optional[Image.Image]]:
    """Crop heads from person crops using YOLO head model.

    Returns a list aligned with `person_crops` where each entry is either the
    head `PIL.Image` or `None` when a head wasn't found for that person.
    """
    images: List[Optional[Image.Image]] = []

    head_class_id = next(
        (k for k, v in model_head.names.items() if v.lower() == "head"),
        None,
    )

    if head_class_id is None:
        # return list of None to preserve alignment
        return [None for _ in person_crops]

    for crop in person_crops:
        if crop is None:
            images.append(None)
            continue

        results = model_head.predict(source=crop, conf=conf, iou=iou, verbose=False)[0]
        boxes = [b for b in results.boxes if int(b.cls) == head_class_id]

        if not boxes:
            images.append(None)
            continue

        best = max(boxes, key=lambda b: float(b.conf))
        x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())

        crop_arr = np.array(crop)
        head = crop_arr[y1:y2, x1:x2]

        if head.size == 0:
            images.append(None)
            continue

        head = _ensure_uint8(head)
        images.append(Image.fromarray(head))

    return images


def crops_deepface_head(
    person_crops: List[Image.Image],
    detector: str = "opencv",
) -> List[Optional[Image.Image]]:
    """Crop heads from person crops using DeepFace face detection.

    Returns a list aligned with `person_crops` where each entry is either the
    head `PIL.Image` or `None` when a face wasn't found for that person.
    """
    from deepface import DeepFace

    images: List[Optional[Image.Image]] = []

    for crop in person_crops:
        if crop is None:
            images.append(None)
            continue

        crop_rgb = np.array(crop)
        try:
            heads = DeepFace.extract_faces(
                img_path=crop_rgb,
                detector_backend=detector,
                align=True,
                enforce_detection=False,
            )
        except Exception:
            images.append(None)
            continue

        if not heads:
            images.append(None)
            continue

        face = heads[0]["face"]
        face = _ensure_uint8(face)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        images.append(Image.fromarray(face))

    return images


# ── LLM attribute detection ─────────────────────────────────

_parser = JsonOutputParser(pydantic_object=ImageAccessoryCounts)


def attribute_calls(
    mode: str,
    question: str,
    image_urls: List[str],
    image_meta: List[str],
    person_summaries: List[str],
    llm: ChatOpenAI,
    feature: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Send labeled crops (for a single source image) to the LLM.

    - `image_urls` and `image_meta` are parallel lists describing each provided crop.
      Example meta: "person_0|head" or "person_1|fullbody".
    - `person_summaries` contains one-line summaries per person describing which
      crops are present/missing (helps LLM be deterministic).

    Returns parsed dict with `per_person` list and aggregated `counts`.
    """
    format_instructions = _parser.get_format_instructions()

    system_prompt = f"""
        You are a deterministic vision analysis system.
        You will be given labeled crops originating from a single source image.
        Each crop is labeled with metadata in the form "person_<i>|<role>" where
        role is one of: "head", "fullbody".

        Use the metadata to associate detections with persons. Do NOT guess
        information that is not visible; if the required view is missing or
        occluded mark the feature as -1 (unknown).

        You are give images : {mode}
        You are given a feature: {feature}

        Decide whether the user's question is about observable human features.
        - If previous feature is not None, you must use it.
        - if previous feature is None, You must extract feature from user input intent, for example "Is this person fat or thin?" -> feature: "fat" , "thin", from user question.
        - If NOT about human observable features OR not answerable from the crops:
            output JSON with status=404 and reason (str) explaining why, e.g:
            "Question is not about observable human features" or any other reason.
            Do NOT output per_person/counts on 404.
        - If NOT observable, return status=404 and empty results.
        - If observable, return status=200 and provide `per_person` results.
        - If question is about opinion or preference use your own opinion, even if its about sensitive topic like ugly, handsome, attractiveness!!!, example: "Is this person attractive?", "Is this person Handsome", etc -> extract the feature also! and give your opinion about it. 
        ### Output format
        Output requirements:
        - `per_person` must include one object per person (ordered by person_id)
          with `person_id` (e.g. "person_0") and `counts` mapping requested
          feature -> 1/0/-1.
        - `counts` at the top level should be the aggregated sum across persons
          where a feature is counted only when value == 1.
        - Use -1 for unknown/occluded and 0 for explicitly absent.

        Be concise and deterministic. {format_instructions}
    """

    # Build human message: include user question, person summaries, then each
    # labeled crop (metadata text immediately before its image_url).
    human_content: List[Dict[str, Any]] = [{"type": "text", "text": question}]

    # provide an explicit person summary to make associations reliable
    human_content.append({"type": "text", "text": "Person summary (crop presence):"})
    for s in person_summaries:
        human_content.append({"type": "text", "text": s})

    human_content.append({"type": "text", "text": "Labeled crops (label before image):"})
    for meta, url in zip(image_meta, image_urls):
        human_content.append({"type": "text", "text": f"Label: {meta}"})
        human_content.append({"type": "image_url", "image_url": {"url": url}})

    resp = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
    ])

    return _parser.parse(resp.content)


# ── Full pipeline ────────────────────────────────────────────

def run_pipeline(
    images: List[Image.Image],
    filenames: List[str],
    mode: str,
    question: str,
    seg_model,
    head_model,
    llm: ChatOpenAI,
) -> List[Dict[str, Any]]:
    """
    Run the full pipeline on a list of images.

    Args:
        images:     list of PIL Images (uploaded/captured)
        filenames:  list of corresponding filenames
        mode:       "Fullbody" | "Head and Fullbody" | "Head"
        question:   user's question/feature query
        seg_model:  YOLOv8m-seg model
        head_model: YOLO head detection model
        llm:        LangChain ChatOpenAI instance

    Returns:
        list of dicts: [{filename, status, counts}, ...]
    """
    results: List[Dict[str, Any]] = []

    for img, fname in zip(images, filenames):
        # Step 1: Segment persons
        feature = None
        person_crops = extract_person_crops(img, seg_model)

        if not person_crops:
            results.append({
                "filename": fname,
                "status": 404,
                "counts": {},
                "error": "No person detected",
            })
            continue

        # Step 2: Head crops aligned to person_crops (preserve index mapping)
        head_crops: List[Optional[Image.Image]] = [None] * len(person_crops)
        if mode in ("Head", "Head and Fullbody"):
            # try YOLO head detector first (returns aligned list)
            head_crops = crops_yolo_head(person_crops, head_model)

            # if some heads still missing, try DeepFace only to fill gaps
            if any(h is None for h in head_crops):
                df_heads = crops_deepface_head(person_crops)
                # fill missing entries from deepface
                head_crops = [h if h is not None else df_h for h, df_h in zip(head_crops, df_heads)]

        # Step 3: Build labeled crop list (one LLM call per source image)
        image_urls: List[str] = []
        image_meta: List[str] = []  # e.g. "person_0|head", "person_0|fullbody"
        person_summaries: List[str] = []

        for idx, person_crop in enumerate(person_crops):
            head = head_crops[idx] if idx < len(head_crops) else None
            head_present = head is not None
            full_present = person_crop is not None

            person_summaries.append(
                f"person_{idx}: head={'present' if head_present else 'missing'}, fullbody={'present' if full_present else 'missing'}"
            )

            if mode == "Fullbody":
                image_urls.append(pil_to_data_url(person_crop))
                image_meta.append(f"person_{idx}|fullbody")
            elif mode == "Head":
                if head_present:
                    image_urls.append(pil_to_data_url(head))
                    image_meta.append(f"person_{idx}|head")
                # if head missing in Head-only mode we do NOT add an image — LLM will be
                # informed via person_summaries and must mark unknown (-1)
            else:  # Head and Fullbody
                if head_present:
                    image_urls.append(pil_to_data_url(head))
                    image_meta.append(f"person_{idx}|head")
                image_urls.append(pil_to_data_url(person_crop))
                image_meta.append(f"person_{idx}|fullbody")

        if not image_urls and mode == "Head":
            results.append({
                "filename": fname,
                "status": 404,
                "counts": {},
                "error": "No head crops available for any person",
            })
            continue

        # Step 4: Single LLM call for the whole image using crop metadata
        try:
            llm_mode = mode.lower().replace("and", "&")
            raw_res = attribute_calls(llm_mode, question, image_urls, image_meta, person_summaries, llm, feature)
            if feature is None and raw_res.get("status") == 200:
                feature = raw_res.get("counts", {}).keys()
            # raw_res should contain `per_person` and optionally aggregated `counts`
            if raw_res.get("status") == 404:
                results.append({
                    "filename": fname,
                    "status": 404,
                    "counts": {},
                    "reason": raw_res.get("reason", ""),
                })
                break
            
            per_person = raw_res.get("per_person", [])

            # determine feature list from LLM output (fallback: empty)
            feature_keys = set()
            for p in per_person:
                feature_keys.update(p.get("counts", {}).keys())
            feature_keys = sorted(feature_keys)

            # ensure every person has an entry; if LLM omitted a person, mark features as -1
            per_person_map = {p["person_id"]: p for p in per_person}
            completed_per_person: List[Dict[str, Any]] = []
            aggregated_counts: Dict[str, int] = {k: 0 for k in feature_keys}

            for idx in range(len(person_crops)):
                pid = f"person_{idx}"
                if pid in per_person_map:
                    entry = per_person_map[pid]
                else:
                    # LLM didn't return this person — mark unknown for each feature
                    entry = {"person_id": pid, "counts": {k: -1 for k in feature_keys}}

                # normalize counts and accumulate only positives
                for k, v in entry.get("counts", {}).items():
                    try:
                        if int(v) == 1:
                            aggregated_counts[k] = aggregated_counts.get(k, 0) + 1
                    except Exception:
                        # ignore invalid values
                        pass

                completed_per_person.append(entry)

            results.append({
                "filename": fname,
                "status": 200,
                "counts": aggregated_counts,
                "per_person": completed_per_person,
            })

        except Exception as e:
            results.append({
                "filename": fname,
                "status": 500,
                "counts": {},
                "error": str(e),
            })

    return results
