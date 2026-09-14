from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..engine import create_ocr, result_data
from ..frames import PROJECT_ROOT, resolve_project_path
from .common import LineupResolutionError


LOCAL_CACHE_DIR = PROJECT_ROOT / ".cache" / "paddlex"
PIXEL_GEOMETRY_COLUMNS = {"x1", "x2", "y1", "y2", "center_x", "center_y"}
NUMBER_SOURCE_COLUMN = "_number_source"


def create_local_table_ocr():
    return create_ocr(LOCAL_CACHE_DIR, recognition_batch_size=8)


def create_local_number_recognizer():
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(LOCAL_CACHE_DIR))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    try:
        from paddleocr import TextRecognition
    except ModuleNotFoundError as exc:
        raise LineupResolutionError("PaddleOCR text recognition is not installed.") from exc
    return TextRecognition(model_name="PP-OCRv6_small_rec", device="cpu")


def require_cv2(purpose: str):
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise LineupResolutionError(f"OpenCV is required for local {purpose} OCR.") from exc
    return cv2


def load_frame_image(frame_path: object, cv2: object) -> np.ndarray | None:
    path = resolve_project_path(Path(str(frame_path)))
    if not path.is_file():
        return None
    return cv2.imread(str(path))


def numeric_observation(
    source_row: pd.Series, *, shirt_number: int, score: float,
    center_x: float, center_y: float, image_width: int, image_height: int,
    half_width: float, half_height: float, include_pixel_geometry: bool,
) -> dict[str, object]:
    output = source_row.to_dict()
    output.update(
        {
            "text": str(shirt_number), "text_type": "shirt_number_candidate",
            "score": round(score, 6), "center_x_norm": round(center_x, 6),
            "center_y_norm": round(center_y, 6),
            NUMBER_SOURCE_COLUMN: "local_ocr",
        }
    )
    if not include_pixel_geometry:
        return output
    center_x_pixels = center_x * image_width
    center_y_pixels = center_y * image_height
    output.update(
        {
            "x1": round(center_x_pixels - half_width), "x2": round(center_x_pixels + half_width),
            "y1": round(center_y_pixels - half_height), "y2": round(center_y_pixels + half_height),
            "center_x": round(center_x_pixels, 3), "center_y": round(center_y_pixels, 3),
        }
    )
    return output


def append_refinement_rows(
    segment: pd.DataFrame, rows: list[dict[str, object]],
) -> tuple[pd.DataFrame, int]:
    if not rows:
        return segment, 0
    base = segment.copy()
    if NUMBER_SOURCE_COLUMN not in base.columns:
        base[NUMBER_SOURCE_COLUMN] = "detector"
    refined = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
    return refined, len(rows)


def numeric_detection(
    result: object, minimum_score: float = 0.75,
) -> tuple[int, float] | None:
    data = result_data(result, LineupResolutionError, "Local PaddleOCR")
    texts = data.get("rec_texts", [])
    scores = data.get("rec_scores", [])
    numeric = [
        (int(str(text).strip()), float(score))
        for text, score in zip(texts, scores, strict=False)
        if str(text).strip().isdigit()
        and 1 <= int(str(text).strip()) <= 99
        and float(score) >= minimum_score
    ]
    return max(numeric, key=lambda item: item[1]) if numeric else None


def recognition_numeric_detection(
    result: object, minimum_score: float = 0.70,
) -> tuple[int, float] | None:
    data = result_data(result, LineupResolutionError, "Local PaddleOCR")
    text = str(data.get("rec_text", "")).strip()
    score = float(data.get("rec_score", 0.0))
    if text.isdigit() and 1 <= int(text) <= 99 and score >= minimum_score:
        return int(text), score
    return None


def number_preprocessing_variants(crop: np.ndarray) -> list[np.ndarray]:
    import cv2
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, inverted = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    monochrome = [gray, clahe, otsu, inverted]
    return [crop, *[cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) for image in monochrome]]


def numeric_candidate_consensus(
    candidates: list[tuple[int, float] | None],
) -> tuple[int, float] | None:
    by_number: defaultdict[int, list[float]] = defaultdict(list)
    for candidate in candidates:
        if candidate is None:
            continue
        number, score = candidate
        by_number[number].append(score)
    if not by_number:
        return None
    winner = max(
        by_number,
        key=lambda number: (
            len(by_number[number]),
            sum(by_number[number]),
            max(by_number[number]),
        ),
    )
    return winner, float(np.mean(by_number[winner]))


def resized_number_crop(
    image: np.ndarray, center_x: float, center_y: float,
    x_radius: float, y_radius: float, scale: int, round_coordinates: bool = False,
) -> np.ndarray:
    import cv2
    height, width = image.shape[:2]
    convert = round if round_coordinates else int
    x1 = max(0, convert((center_x - x_radius) * width))
    x2 = min(width, convert((center_x + x_radius) * width))
    y1 = max(0, convert((center_y - y_radius) * height))
    y2 = min(height, convert((center_y + y_radius) * height))
    return cv2.resize(
        image[y1:y2, x1:x2], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
    )
