from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING, Sequence
import cv2
import numpy as np
if TYPE_CHECKING:
    from paddleocr import PaddleOCR

@functools.lru_cache(maxsize=2)
def get_ocr_model(det_model: str = "PP-OCRv6_tiny_det", rec_model: str = "PP-OCRv6_tiny_rec") -> PaddleOCR:
    """Load on demand; retain both the detector's Tiny and extractor's Small models."""
    from paddleocr import PaddleOCR

    return PaddleOCR(
        text_detection_model_name=det_model,
        text_recognition_model_name=rec_model,
        device="cpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def extract_frames_at_timestamps(
    video_path: Path,
    timestamps: Sequence[float],
) -> tuple[list[float], list[np.ndarray]]:
    """Seek and extract frames efficiently from video at given timestamps."""
    cap = cv2.VideoCapture(str(video_path))
    valid_ts: list[float] = []
    frames: list[np.ndarray] = []

    for t in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ret, frame = cap.read()
        if ret and frame is not None:
            valid_ts.append(t)
            frames.append(frame)

    cap.release()
    return valid_ts, frames


def run_ocr_batch(
    frames: list[np.ndarray],
    ocr_model: PaddleOCR,
) -> list[tuple[list[str], list[float], list]]:
    """Run batch OCR prediction and return parsed texts, confidence scores, and boxes."""
    if not frames:
        return []

    results = list(ocr_model.predict(frames))
    parsed: list[tuple[list[str], list[float], list]] = []

    for res in results:
        payload = res.json if isinstance(res.json, dict) else res.json()
        data = payload.get("res", payload)
        texts = data.get("rec_texts", [])
        scores = data.get("rec_scores", [])
        # Recognition may filter detections. Only these polygons align with rec_texts.
        boxes = data.get("rec_polys")
        if boxes is None:
            boxes = data.get("dt_polys", [])
        if not len(texts) == len(scores) == len(boxes):
            raise ValueError("OCR text/confidence/polygon counts do not match")
        parsed.append((texts, scores, boxes))

    return parsed
