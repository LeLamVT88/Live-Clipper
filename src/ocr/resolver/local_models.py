"""Models and image helpers used by targeted local OCR."""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np

from .common import PROJECT_ROOT, LineupResolutionError


def create_local_table_ocr():
    from ..ocr_engine import create_ocr

    return create_ocr(
        PROJECT_ROOT / ".cache" / "paddlex",
        recognition_batch_size=8,
    )


def create_local_number_recognizer():
    os.environ.setdefault(
        "PADDLE_PDX_CACHE_HOME",
        str(PROJECT_ROOT / ".cache" / "paddlex"),
    )
    os.environ.setdefault(
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK",
        "True",
    )
    try:
        from paddleocr import TextRecognition
    except ModuleNotFoundError as exc:
        raise LineupResolutionError(
            "PaddleOCR text recognition is not installed."
        ) from exc
    return TextRecognition(
        model_name="PP-OCRv6_small_rec",
        device="cpu",
    )


def local_result_data(result: object) -> dict[str, object]:
    payload = getattr(result, "json", None)
    if callable(payload):
        payload = payload()
    if not isinstance(payload, dict):
        raise LineupResolutionError(
            "Local PaddleOCR returned an unsupported result format."
        )
    data = payload.get("res", payload)
    if not isinstance(data, dict):
        raise LineupResolutionError(
            "Local PaddleOCR result does not contain a result object."
        )
    return data


def numeric_detection(
    result: object,
    minimum_score: float = 0.75,
) -> tuple[int, float] | None:
    data = local_result_data(result)
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
    result: object,
    minimum_score: float = 0.70,
) -> tuple[int, float] | None:
    data = local_result_data(result)
    text = str(data.get("rec_text", "")).strip()
    score = float(data.get("rec_score", 0.0))
    if (
        text.isdigit()
        and 1 <= int(text) <= 99
        and score >= minimum_score
    ):
        return int(text), score
    return None


def number_preprocessing_variants(
    crop: np.ndarray,
) -> list[np.ndarray]:
    import cv2

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(4, 4),
    ).apply(gray)
    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    _, inverted = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    monochrome = [gray, clahe, otsu, inverted]
    return [
        crop,
        *[
            cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for image in monochrome
        ],
    ]


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
    image: np.ndarray,
    center_x: float,
    center_y: float,
    x_radius: float,
    y_radius: float,
    scale: int,
    round_coordinates: bool = False,
) -> np.ndarray:
    import cv2

    height, width = image.shape[:2]
    convert = (
        lambda value: int(round(value))
        if round_coordinates
        else int(value)
    )
    x1 = max(0, convert((center_x - x_radius) * width))
    x2 = min(width, convert((center_x + x_radius) * width))
    y1 = max(0, convert((center_y - y_radius) * height))
    y2 = min(height, convert((center_y + y_radius) * height))
    return cv2.resize(
        image[y1:y2, x1:x2],
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )
