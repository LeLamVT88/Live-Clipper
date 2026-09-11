from __future__ import annotations

from collections import Counter
from statistics import median

import cv2
import numpy as np

from mapping.schema import BBox, FrameAnalysis, PlayerObservation
from mapping.text import parse_number_crop
from ocr_engine import run_ocr_batch


def _fallback_search_box(observation: PlayerObservation) -> BBox:
    x0, y0, x1, y1 = observation.name_box
    if observation.panel_role == "starter_list":
        return max(0.0, x0 - .16), max(0.0, y0 - .018), max(0.0, x0 - .003), min(1.0, y1 + .018)
    center = (x0 + x1) / 2
    half_width = max(.032, min(.055, (x1 - x0) * .60))
    return (max(0.0, center - half_width), max(0.0, y0 - .10),
            min(1.0, center + half_width), max(0.0, y0 - .008))


def _search_boxes(observation: PlayerObservation, anchors: list[PlayerObservation]) -> list[BBox]:
    boxes: list[BBox] = []
    usable = [anchor for anchor in anchors if anchor.panel_role == observation.panel_role
              and anchor.number_box is not None and anchor.number_source != "inline_ocr"]
    if len(usable) >= 2:
        relative_x = []
        relative_y = []
        widths = []
        heights = []
        for anchor in usable:
            name_x = (anchor.name_box[0] + anchor.name_box[2]) / 2
            name_y = (anchor.name_box[1] + anchor.name_box[3]) / 2
            number_x = (anchor.number_box[0] + anchor.number_box[2]) / 2
            number_y = (anchor.number_box[1] + anchor.number_box[3]) / 2
            relative_x.append(number_x - name_x)
            relative_y.append(number_y - name_y)
            widths.append(anchor.number_box[2] - anchor.number_box[0])
            heights.append(anchor.number_box[3] - anchor.number_box[1])
        name_x = (observation.name_box[0] + observation.name_box[2]) / 2
        name_y = (observation.name_box[1] + observation.name_box[3]) / 2
        center_x = name_x + median(relative_x)
        center_y = name_y + median(relative_y)
        half_width = max(.018, min(.055, median(widths) * 1.8))
        half_height = max(.022, min(.065, median(heights) * 1.4))
        boxes.append((max(0.0, center_x - half_width), max(0.0, center_y - half_height),
                      min(1.0, center_x + half_width), min(1.0, center_y + half_height)))
    fallback = _fallback_search_box(observation)
    if not boxes or max(abs(a - b) for a, b in zip(boxes[0], fallback)) > .015:
        boxes.append(fallback)
    return boxes


def _crop(frame: np.ndarray, box: BBox) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = (
        max(0, min(width, round(box[0] * width))),
        max(0, min(height, round(box[1] * height))),
        max(0, min(width, round(box[2] * width))),
        max(0, min(height, round(box[3] * height))),
    )
    patch = frame[y0:y1, x0:x1]
    return patch if patch.size and patch.shape[0] >= 4 and patch.shape[1] >= 4 else None


def _variants(patch: np.ndarray) -> list[np.ndarray]:
    scale = max(1.0, 120 / max(1, patch.shape[0]))
    enlarged = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [
        enlarged,
        cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(255 - otsu, cv2.COLOR_GRAY2BGR),
    ]


def recover_frame_numbers(frame: np.ndarray, analysis: FrameAnalysis, model: object) -> list[dict[str, object]]:
    """Recover missing digits locally; variants support one timestamp, not extra votes."""
    anchors = [observation for observation in analysis.starter_observations
               if observation.jersey_number is not None and observation.number_box is not None]
    pending: list[tuple[PlayerObservation, list[BBox], list[np.ndarray]]] = []
    batch: list[np.ndarray] = []
    for observation in analysis.starter_observations:
        if observation.jersey_number is not None:
            continue
        boxes = _search_boxes(observation, anchors)
        variants = []
        valid_boxes = []
        for box in boxes:
            patch = _crop(frame, box)
            if patch is None:
                continue
            valid_boxes.append(box)
            variants.extend(_variants(patch))
        if variants:
            pending.append((observation, valid_boxes, variants))
            batch.extend(variants)
    if not batch:
        return []

    results = run_ocr_batch(batch, model)
    cursor = 0
    evidence = []
    for observation, boxes, variants in pending:
        readings = []
        votes: Counter[int] = Counter()
        for variant_index in range(len(variants)):
            texts, scores, _ = results[cursor]
            cursor += 1
            values = sorted({number for text in texts if (number := parse_number_crop(text)) is not None})
            readings.append({"crop": variant_index // 4, "variant": variant_index % 4, "texts": texts,
                             "confidences": [round(float(score), 4) for score in scores], "numbers": values})
            if len(values) == 1:
                votes[values[0]] += 1
        candidates = sorted(votes)
        observation.number_candidates = sorted(set(observation.number_candidates) | set(candidates))
        if len(candidates) == 1 and votes[candidates[0]] >= 2:
            observation.jersey_number = candidates[0]
            observation.number_confidence = votes[candidates[0]] / len(variants)
            observation.number_box = boxes[0]
            observation.number_source = "local_preprocessed_ocr"
            observation.pair_confidence = observation.number_confidence * .8
        evidence.append({
            "timestamp": round(analysis.timestamp, 3), "name": observation.name,
            "search_box": [round(value, 5) for value in boxes[0]],
            "search_boxes": [[round(value, 5) for value in box] for box in boxes],
            "variant_readings": readings,
            "accepted_number": observation.jersey_number if observation.number_source == "local_preprocessed_ocr" else None,
        })
    return evidence
