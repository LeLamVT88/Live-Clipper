"""Tiny PaddleOCR worker used by visual_refinement.py.

This file intentionally has no imports from the main environment. It runs in
``.venv-ocr`` so Qwen/PySceneDetect and Paddle dependencies stay isolated.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DETECTION_MODEL = "PP-OCRv6_tiny_det"
RECOGNITION_MODEL = "PP-OCRv6_tiny_rec"
FRAME_WIDTH = 960
DETECTION_BATCH_SIZE = 8
RECOGNITION_BATCH_SIZE = 8
MIN_DETECTION_SCORE = 0.45
MIN_RECOGNITION_SCORE = 0.55
MIN_SHORTLIST_BOXES = 5
MAX_SAVED_TEXTS = 20
SHORTLIST_BUCKET_SECONDS = 6.0
MAX_GRAPHIC_SECONDS = 60.0
MAX_WEAK_BRIDGE_SECONDS = 4.0
MAX_SHORT_SCENE_SECONDS = 18.0
MAX_POSITIVE_GAP_SECONDS = 15.0
LAYOUT_GRID_COLUMNS = 8
LAYOUT_GRID_ROWS = 6
APPEARANCE_H_BINS = 12
APPEARANCE_S_BINS = 8
MIN_SCENE_APPEARANCE_SIMILARITY = 0.8
IDENTITY_TEXT_EXCLUSIONS = frozenset(
    {
        "champions",
        "elite",
        "fifa",
        "fifa com",
        "finals",
        "formation",
        "gk",
        "league",
        "lineup",
        "live",
        "starting eleven",
        "substitutes",
        "tv360",
    }
)
FORMATION_PATTERN = re.compile(r"\b[1-5](?:\s*[-–]\s*[1-5]){2,3}\b")
NUMBER_PATTERN = re.compile(r"(?<!\d)(?:[1-9]|[1-9]\d)(?!\d)")
LINEUP_TERMS = (
    "lineup",
    "line up",
    "starting xi",
    "starting eleven",
    "starters",
    "substitutes",
    "substitution",
    "replacements",
    "formation",
    "doi hinh",
    "xuất phát",
    "xuat phat",
    "formazione",
    "suplentes",
    "titulares",
)
NON_LINEUP_TERMS = (
    "match officials",
    "match official",
    "referee",
    "video assistant referee",
    "video match officials",
)
class OCRWorkerError(RuntimeError):
    pass


@dataclass(frozen=True)
class Unit:
    index: int
    start_seconds: float
    end_seconds: float
    sample_seconds: float
    scene_index: int = 0


@dataclass(frozen=True)
class Detection:
    box_count: int
    area_ratio: float
    x_span: float = 0.0
    y_span: float = 0.0
    layout_cells: tuple[int, ...] = ()
    appearance_histogram: tuple[float, ...] = ()

    @property
    def density(self) -> float:
        return min(1.0, self.box_count / 14) * 0.8 + min(
            1.0, self.area_ratio / 0.12
        ) * 0.2

    @property
    def lineup_layout(self) -> bool:
        """Cheap shape gate for a roster column or formation graphic."""
        return self.area_ratio >= 0.15 or (
            self.box_count >= 3
            and (
                self.y_span >= 0.24
                or self.area_ratio >= 0.018
                or (self.box_count >= 6 and self.x_span >= 0.25)
            )
        )


@dataclass(frozen=True)
class Recognition:
    positive: bool
    score: float
    texts: tuple[str, ...]


def _result_data(result: object) -> dict[str, object]:
    payload = getattr(result, "json", None)
    if callable(payload):
        payload = payload()
    if not isinstance(payload, dict):
        raise OCRWorkerError("PaddleOCR returned an unsupported result format.")
    data = payload.get("res", payload)
    if not isinstance(data, dict):
        raise OCRWorkerError("PaddleOCR result has no result object.")
    return data


def _as_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        return list(value)  # type: ignore[arg-type]
    except TypeError:
        return []


class VideoFrames:
    def __init__(self, video_path: Path) -> None:
        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise OCRWorkerError("opencv-python is not installed.") from exc
        self.cv2 = cv2
        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            raise OCRWorkerError(f"Cannot open video: {video_path}")

    def read(self, seconds: float):
        self.capture.set(self.cv2.CAP_PROP_POS_MSEC, max(0.0, seconds) * 1000)
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise OCRWorkerError(f"Cannot read video frame at {seconds:.3f}s.")
        height, width = frame.shape[:2]
        if width > FRAME_WIDTH:
            scale = FRAME_WIDTH / width
            frame = self.cv2.resize(
                frame,
                (FRAME_WIDTH, max(1, round(height * scale))),
                interpolation=self.cv2.INTER_AREA,
            )
        return frame

    def close(self) -> None:
        self.capture.release()


def _load_units(raw_task: dict[str, object]) -> tuple[Unit, ...]:
    raw_units = raw_task.get("units", [])
    if not isinstance(raw_units, list):
        raise OCRWorkerError("Task units must be an array.")
    units: list[Unit] = []
    for index, raw_unit in enumerate(raw_units):
        if not isinstance(raw_unit, dict):
            raise OCRWorkerError("Invalid visual unit.")
        try:
            start = float(raw_unit["start_seconds"])
            end = float(raw_unit["end_seconds"])
            sample = float(raw_unit["sample_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise OCRWorkerError("Invalid visual unit values.") from exc
        if start < 0 or end <= start or not start <= sample <= end:
            raise OCRWorkerError("Invalid visual unit range.")
        try:
            scene_index = int(raw_unit.get("scene_index", index))
        except (TypeError, ValueError) as exc:
            raise OCRWorkerError("Invalid visual scene index.") from exc
        if scene_index < 0:
            raise OCRWorkerError("Visual scene index cannot be negative.")
        units.append(Unit(index, start, end, sample, scene_index))
    return tuple(units)


def _middle_out(units: Sequence[Unit], center_seconds: float) -> tuple[int, ...]:
    """Visit the raw candidate center first, alternating toward both edges."""
    return tuple(
        unit.index
        for unit in sorted(
            units,
            key=lambda item: (
                abs(item.sample_seconds - center_seconds),
                item.sample_seconds,
            ),
        )
    )


def _batches(values: Sequence[int], size: int) -> Iterable[Sequence[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _detect_frames(
    model: object,
    reader: VideoFrames,
    units: Sequence[Unit],
    order: Sequence[int],
) -> dict[int, Detection]:
    detections: dict[int, Detection] = {}
    for batch_indices in _batches(order, DETECTION_BATCH_SIZE):
        batch_frames = [
            reader.read(units[index].sample_seconds) for index in batch_indices
        ]
        results = list(model.predict(batch_frames, batch_size=DETECTION_BATCH_SIZE))
        if len(results) != len(batch_indices):
            raise OCRWorkerError("TextDetection returned an unexpected batch size.")
        for index, frame, result in zip(
            batch_indices, batch_frames, results, strict=True
        ):
            data = _result_data(result)
            raw_scores = _as_list(data.get("dt_scores"))
            raw_polygons = _as_list(data.get("dt_polys"))
            height, width = frame.shape[:2]
            image_area = max(1.0, float(width * height))
            accepted_boxes: list[tuple[float, float, float, float]] = []
            for score, polygon in zip(raw_scores, raw_polygons):
                if float(score) < MIN_DETECTION_SCORE:
                    continue
                points = [_as_list(point) for point in _as_list(polygon)]
                xs = [float(point[0]) for point in points if len(point) >= 2]
                ys = [float(point[1]) for point in points if len(point) >= 2]
                if xs and ys:
                    accepted_boxes.append(
                        (
                            max(0.0, min(xs)),
                            max(0.0, min(ys)),
                            min(float(width), max(xs)),
                            min(float(height), max(ys)),
                        )
                    )
            accepted_areas = [
                max(0.0, right - left) * max(0.0, bottom - top)
                for left, top, right, bottom in accepted_boxes
            ]
            layout_cells = {
                min(
                    LAYOUT_GRID_ROWS - 1,
                    int(((top + bottom) / 2) / max(1, height) * LAYOUT_GRID_ROWS),
                )
                * LAYOUT_GRID_COLUMNS
                + min(
                    LAYOUT_GRID_COLUMNS - 1,
                    int(((left + right) / 2) / max(1, width) * LAYOUT_GRID_COLUMNS),
                )
                for left, top, right, bottom in accepted_boxes
            }
            frame_height, frame_width = frame.shape[:2]
            half_height = frame_height // 2
            half_width = frame_width // 2
            appearance_regions = (
                frame,
                frame[:, :half_width],
                frame[:, half_width:],
                frame[:half_height, :half_width],
                frame[half_height:, :half_width],
                frame[:half_height, half_width:],
                frame[half_height:, half_width:],
            )
            appearance_values: list[float] = []
            for region in appearance_regions:
                hsv_region = reader.cv2.cvtColor(
                    region, reader.cv2.COLOR_BGR2HSV
                )
                region_histogram = reader.cv2.calcHist(
                    [hsv_region],
                    [0, 1],
                    None,
                    [APPEARANCE_H_BINS, APPEARANCE_S_BINS],
                    [0, 180, 0, 256],
                )
                region_norm = float(
                    reader.cv2.norm(region_histogram, reader.cv2.NORM_L2)
                )
                if region_norm > 0:
                    region_histogram /= region_norm
                appearance_values.extend(
                    float(value) for value in region_histogram.reshape(-1)
                )
            detections[index] = Detection(
                box_count=len(accepted_areas),
                area_ratio=sum(accepted_areas) / image_area,
                x_span=(
                    (
                        max(box[2] for box in accepted_boxes)
                        - min(box[0] for box in accepted_boxes)
                    )
                    / max(1, width)
                    if accepted_boxes
                    else 0.0
                ),
                y_span=(
                    (
                        max(box[3] for box in accepted_boxes)
                        - min(box[1] for box in accepted_boxes)
                    )
                    / max(1, height)
                    if accepted_boxes
                    else 0.0
                ),
                layout_cells=tuple(sorted(layout_cells)),
                appearance_histogram=tuple(appearance_values),
            )
    return detections


def _shortlist(
    detections: dict[int, Detection],
    *,
    units: Sequence[Unit],
    center_order: Sequence[int],
    limit: int,
) -> tuple[int, ...]:
    """Select dense frames plus neighbors; recognition never covers all 600s."""
    center_rank = {index: rank for rank, index in enumerate(center_order)}
    eligible = sorted(
        (
            index
            for index, detection in detections.items()
            if detection.box_count >= MIN_SHORTLIST_BOXES
        ),
        key=lambda index: (-detections[index].density, center_rank[index]),
    )
    bucket_seeds: dict[int, int] = {}
    for index in eligible:
        bucket = math.floor(units[index].sample_seconds / SHORTLIST_BUCKET_SECONDS)
        bucket_seeds.setdefault(bucket, index)
    seeds = sorted(
        bucket_seeds.values(),
        key=lambda index: (-detections[index].density, center_rank[index]),
    )[:limit]
    # Cover as many time buckets as the budget permits before spending the
    # remaining recognition slots on neighboring frames.  This lets a dense
    # non-lineup graphic just outside the roster become a semantic stop signal.
    selected: list[int] = list(seeds)
    if len(selected) >= limit:
        return tuple(sorted(selected[:limit]))
    for seed in seeds:
        for index in (seed - 1, seed + 1):
            if index in detections and index not in selected:
                selected.append(index)
                if len(selected) >= limit:
                    return tuple(sorted(selected))
    for seed in eligible:
        if seed not in selected:
            selected.append(seed)
            if len(selected) >= limit:
                break
    return tuple(sorted(selected))


def _recognition_score(texts: Sequence[str], detection: Detection) -> Recognition:
    normalized = " ".join(text.casefold() for text in texts)
    formation = bool(FORMATION_PATTERN.search(normalized))
    keyword = any(term in normalized for term in LINEUP_TERMS)
    number_count = len(NUMBER_PATTERN.findall(normalized))
    alpha_count = sum(
        1
        for text in texts
        if sum(character.isalpha() for character in text) >= 3
    )
    score = (
        (0.28 if formation else 0.0)
        + (0.24 if keyword else 0.0)
        + min(1.0, number_count / 7) * 0.25
        + min(1.0, alpha_count / 8) * 0.16
        + detection.density * 0.15
    )
    structural = number_count >= 5 and alpha_count >= 5
    semantic = (formation or keyword) and detection.box_count >= 5
    positive = score >= 0.55 and (structural or semantic)
    return Recognition(
        positive,
        min(1.0, score),
        tuple(texts[:MAX_SAVED_TEXTS]),
    )


def _recognize_frames(
    model: object,
    reader: VideoFrames,
    units: Sequence[Unit],
    indices: Sequence[int],
    detections: dict[int, Detection],
) -> dict[int, Recognition]:
    recognitions: dict[int, Recognition] = {}
    for batch_indices in _batches(indices, RECOGNITION_BATCH_SIZE):
        batch_frames = [
            reader.read(units[index].sample_seconds) for index in batch_indices
        ]
        results = list(model.predict(batch_frames))
        if len(results) != len(batch_indices):
            raise OCRWorkerError("PaddleOCR returned an unexpected batch size.")
        for index, result in zip(batch_indices, results, strict=True):
            data = _result_data(result)
            raw_texts = _as_list(data.get("rec_texts"))
            raw_scores = _as_list(data.get("rec_scores"))
            texts = [
                str(text).strip()
                for text, score in zip(raw_texts, raw_scores)
                if str(text).strip() and float(score) >= MIN_RECOGNITION_SCORE
            ]
            recognitions[index] = _recognition_score(texts, detections[index])
    return recognitions


def _layout_similarity(first: Detection, second: Detection) -> float:
    first_cells = set(first.layout_cells)
    second_cells = set(second.layout_cells)
    if not first_cells or not second_cells:
        return 0.0
    return 2 * len(first_cells & second_cells) / (
        len(first_cells) + len(second_cells)
    )


def _appearance_similarity(first: Detection, second: Detection) -> float:
    if (
        not first.appearance_histogram
        or len(first.appearance_histogram) != len(second.appearance_histogram)
    ):
        return 0.0
    histogram_size = APPEARANCE_H_BINS * APPEARANCE_S_BINS
    if (
        len(first.appearance_histogram) >= histogram_size
        and len(first.appearance_histogram) % histogram_size == 0
    ):
        similarities = (
            sum(
                first.appearance_histogram[offset + index]
                * second.appearance_histogram[offset + index]
                for index in range(histogram_size)
            )
            for offset in range(
                0,
                len(first.appearance_histogram),
                histogram_size,
            )
        )
        similarity = max(similarities, default=0.0)
    else:
        similarity = sum(
            first_value * second_value
            for first_value, second_value in zip(
                first.appearance_histogram,
                second.appearance_histogram,
                strict=True,
            )
        )
    return max(0.0, min(1.0, similarity))


def _matches_lineup_appearance(
    detection: Detection,
    anchors: Sequence[Detection],
) -> bool:
    return any(
        _appearance_similarity(detection, anchor)
        >= MIN_SCENE_APPEARANCE_SIMILARITY
        for anchor in anchors
    )


def _identity_texts(recognition: Recognition) -> set[str]:
    identities: set[str] = set()
    for raw_text in recognition.texts:
        normalized = " ".join(
            "".join(
                character if character.isalnum() else " "
                for character in raw_text.casefold()
            ).split()
        )
        letters = sum(character.isalpha() for character in normalized)
        if (
            3 <= letters <= 32
            and not any(character.isdigit() for character in normalized)
            and normalized not in IDENTITY_TEXT_EXCLUSIONS
            and not any(
                term in normalized
                for term in ("fifa", "champions league", "substitutes")
            )
        ):
            identities.add(normalized)
    return identities


def _shares_lineup_identity(
    recognition: Recognition,
    anchors: Sequence[Recognition],
) -> bool:
    candidate_identities = _identity_texts(recognition)
    return bool(candidate_identities) and any(
        candidate_identities & _identity_texts(anchor)
        for anchor in anchors
    )


def _supports_lineup_context(
    detection: Detection,
    anchors: Sequence[Detection],
) -> bool:
    """Keep visually compatible text layouts without reading every frame."""
    if not detection.lineup_layout:
        return False
    similarities = tuple(
        _layout_similarity(detection, anchor) for anchor in anchors
    )
    if not any(anchor.layout_cells for anchor in anchors):
        return detection.box_count >= 7 or detection.area_ratio >= 0.025
    best_similarity = max(similarities, default=0.0)
    return best_similarity >= 0.3 or (
        best_similarity >= 0.1
        and (detection.box_count >= 12 or detection.area_ratio >= 0.04)
    )


def _context_indices(
    units: Sequence[Unit],
    detections: dict[int, Detection],
    recognitions: dict[int, Recognition],
    positive_indices: Sequence[int],
    barriers: set[int],
) -> set[int]:
    positives = tuple(sorted(positive_indices))
    if not positives:
        return set()
    anchors = tuple(detections[index] for index in positives)
    recognition_anchors = tuple(
        recognitions[index] for index in positives if index in recognitions
    )
    supported = {
        index
        for index, detection in detections.items()
        if index not in barriers
        and (
            _supports_lineup_context(detection, anchors)
            or _matches_lineup_appearance(detection, anchors)
        )
    }
    supported.update(positives)
    supported.update(
        index
        for index, recognition in recognitions.items()
        if index not in barriers
        and _shares_lineup_identity(recognition, recognition_anchors)
    )

    by_scene: dict[int, list[int]] = {}
    for unit in units:
        by_scene.setdefault(unit.scene_index, []).append(unit.index)
    for indices in by_scene.values():
        duration = units[indices[-1]].end_seconds - units[indices[0]].start_seconds
        if duration <= MAX_SHORT_SCENE_SECONDS and any(
            index in supported for index in indices
        ):
            supported.update(index for index in indices if index not in barriers)
    return supported


def _path_is_continuous(
    units: Sequence[Unit],
    supported: set[int],
    start_index: int,
    end_index: int,
    barriers: set[int],
) -> bool:
    if (
        units[end_index].end_seconds - units[start_index].start_seconds
        > MAX_GRAPHIC_SECONDS
    ):
        return False
    weak_seconds = 0.0
    for index in range(start_index, end_index + 1):
        if index in barriers:
            return False
        if index in supported:
            weak_seconds = 0.0
            continue
        weak_seconds += units[index].end_seconds - units[index].start_seconds
        if weak_seconds > MAX_WEAK_BRIDGE_SECONDS:
            return False
    return True


def _event_groups(
    units: Sequence[Unit],
    seed_indices: Sequence[int],
    barriers: set[int],
) -> tuple[tuple[int, ...], ...]:
    positives = sorted(seed_indices)
    if not positives:
        return ()
    groups: list[list[int]] = [[positives[0]]]
    for index in positives[1:]:
        previous = groups[-1][-1]
        positive_gap = (
            units[index].start_seconds - units[previous].end_seconds
        )
        total_duration = (
            units[index].end_seconds
            - units[groups[-1][0]].start_seconds
        )
        if (
            positive_gap <= MAX_POSITIVE_GAP_SECONDS
            and total_duration <= MAX_GRAPHIC_SECONDS
            and not any(
                barrier in range(previous + 1, index)
                for barrier in barriers
            )
        ):
            groups[-1].append(index)
        else:
            groups.append([index])
    return tuple(tuple(group) for group in groups)


def _expand_event_indices(
    units: Sequence[Unit],
    detections: dict[int, Detection],
    recognitions: dict[int, Recognition],
    supported: set[int],
    barriers: set[int],
    group: Sequence[int],
) -> tuple[int, int]:
    left, right = group[0], group[-1]
    anchors = tuple(detections[index] for index in group)
    recognition_anchors = tuple(
        recognitions[index] for index in group if index in recognitions
    )
    while left > 0:
        candidate = left - 1
        if units[candidate].scene_index != units[left].scene_index:
            candidate_scene_index = units[candidate].scene_index
            candidate_scene_units = tuple(
                unit
                for unit in units
                if unit.scene_index == candidate_scene_index
            )
            scene_matches_identity = any(
                unit.index in recognitions
                and _shares_lineup_identity(
                    recognitions[unit.index], recognition_anchors
                )
                for unit in candidate_scene_units
            )
            scene_matches_appearance = any(
                _matches_lineup_appearance(detections[unit.index], anchors)
                for unit in candidate_scene_units
            )
            has_appearance = bool(
                detections[candidate].appearance_histogram
                and any(anchor.appearance_histogram for anchor in anchors)
            )
            if has_appearance and not (
                scene_matches_appearance or scene_matches_identity
            ):
                break
            scene_duration = (
                candidate_scene_units[-1].end_seconds
                - candidate_scene_units[0].start_seconds
            )
            if (
                scene_duration > MAX_SHORT_SCENE_SECONDS
                and detections[candidate].area_ratio < 0.15
                and candidate not in group
                and not (scene_matches_appearance or scene_matches_identity)
            ):
                break
        if not _path_is_continuous(
            units, supported, candidate, right, barriers
        ):
            break
        if candidate in supported:
            left = candidate
            continue
        bridge = candidate
        while bridge >= 0 and bridge not in supported:
            bridge -= 1
        if bridge < 0 or not _path_is_continuous(
            units, supported, bridge, right, barriers
        ):
            break
        left = bridge
    return left, right


def _events(
    task_id: str,
    units: Sequence[Unit],
    detections: dict[int, Detection],
    recognitions: dict[int, Recognition],
    *,
    center_seconds: float,
    max_events: int,
    prefer_center: bool,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    barriers = {
        index
        for index, recognition in recognitions.items()
        if not recognition.positive
        and any(
            term in " ".join(text.casefold() for text in recognition.texts)
            for term in NON_LINEUP_TERMS
        )
    }
    seed_indices = tuple(
        index
        for index, recognition in recognitions.items()
        if recognition.positive
    )
    for group in _event_groups(units, seed_indices, barriers):
        supported = _context_indices(
            units,
            detections,
            recognitions,
            group,
            barriers,
        )
        left, right = _expand_event_indices(
            units,
            detections,
            recognitions,
            supported,
            barriers,
            group,
        )
        recognized = [recognitions[index] for index in group]
        texts = list(dict.fromkeys(text for item in recognized for text in item.texts))
        events.append(
            {
                "task_id": task_id,
                "start_seconds": round(units[left].start_seconds, 3),
                "end_seconds": round(units[right].end_seconds, 3),
                "confidence": round(
                    sum(item.score for item in recognized) / len(recognized), 4
                ),
                "sample_seconds": [
                    round(units[index].sample_seconds, 3) for index in group
                ],
                "texts": texts[:MAX_SAVED_TEXTS],
            }
        )
    unique_events: dict[tuple[float, float], dict[str, object]] = {}
    for event in events:
        key = (float(event["start_seconds"]), float(event["end_seconds"]))
        existing = unique_events.get(key)
        if existing is None or float(event["confidence"]) > float(
            existing["confidence"]
        ):
            unique_events[key] = event
    events = list(unique_events.values())
    if prefer_center:
        events.sort(
            key=lambda event: (
                abs(
                    (
                        float(event["start_seconds"])
                        + float(event["end_seconds"])
                    )
                    / 2
                    - center_seconds
                ),
                -float(event["confidence"]),
            )
        )
    else:
        events.sort(
            key=lambda event: (-float(event["confidence"]), event["start_seconds"])
        )
    return events[:max_events]


def _unit_diagnostics(
    units: Sequence[Unit],
    detections: dict[int, Detection],
    recognitions: dict[int, Recognition],
    selected_indices: Sequence[int],
) -> list[dict[str, object]]:
    selected = set(selected_indices)
    diagnostics: list[dict[str, object]] = []
    for unit in units:
        detection = detections[unit.index]
        recognition = recognitions.get(unit.index)
        item: dict[str, object] = {
            "index": unit.index,
            "scene_index": unit.scene_index,
            "start_seconds": round(unit.start_seconds, 3),
            "end_seconds": round(unit.end_seconds, 3),
            "sample_seconds": round(unit.sample_seconds, 3),
            "text_box_count": detection.box_count,
            "text_area_ratio": round(detection.area_ratio, 6),
            "text_density": round(detection.density, 4),
            "selected_for_recognition": unit.index in selected,
        }
        if recognition is not None:
            item["ocr_positive"] = recognition.positive
            item["ocr_score"] = round(recognition.score, 4)
            item["recognized_texts"] = list(recognition.texts)
        diagnostics.append(item)
    return diagnostics


def _create_detection_model() -> object:
    try:
        from paddleocr import TextDetection
    except ModuleNotFoundError as exc:
        raise OCRWorkerError(
            "PaddleOCR is not installed; run with .venv-ocr/bin/python."
        ) from exc
    return TextDetection(model_name=DETECTION_MODEL, device="cpu")


def _create_recognition_model() -> object:
    try:
        from paddleocr import PaddleOCR
    except ModuleNotFoundError as exc:
        raise OCRWorkerError(
            "PaddleOCR is not installed; run with .venv-ocr/bin/python."
        ) from exc
    return PaddleOCR(
        text_detection_model_name=DETECTION_MODEL,
        text_recognition_model_name=RECOGNITION_MODEL,
        text_recognition_batch_size=RECOGNITION_BATCH_SIZE,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
    )


def run(request: dict[str, object]) -> dict[str, object]:
    video_path = Path(str(request.get("video_path", ""))).expanduser().resolve()
    if not video_path.is_file():
        raise OCRWorkerError(f"Video does not exist: {video_path}")
    raw_tasks = request.get("tasks", [])
    if not isinstance(raw_tasks, list):
        raise OCRWorkerError("tasks must be an array.")
    try:
        recognition_limit = int(request.get("max_recognition_frames", 40))
    except (TypeError, ValueError) as exc:
        raise OCRWorkerError("Invalid recognition frame limit.") from exc
    if recognition_limit <= 0:
        raise OCRWorkerError("Recognition frame limit must be positive.")

    detector = _create_detection_model()
    recognizer: object | None = None
    reader = VideoFrames(video_path)
    task_results: list[dict[str, object]] = []
    total_recognized = 0
    try:
        for raw_task in raw_tasks:
            if not isinstance(raw_task, dict):
                raise OCRWorkerError("Invalid OCR task.")
            task_id = str(raw_task.get("task_id", "")).strip()
            mode = str(raw_task.get("mode", "")).strip()
            center = float(raw_task.get("center_seconds", 0.0))
            max_events = int(raw_task.get("max_events", 1))
            units = _load_units(raw_task)
            order = _middle_out(units, center)
            detections = _detect_frames(detector, reader, units, order)
            shortlist = _shortlist(
                detections,
                units=units,
                center_order=order,
                limit=recognition_limit,
            )
            if shortlist and recognizer is None:
                recognizer = _create_recognition_model()
            recognitions = (
                _recognize_frames(
                    recognizer,
                    reader,
                    units,
                    shortlist,
                    detections,
                )
                if recognizer is not None and shortlist
                else {}
            )
            total_recognized += len(shortlist)
            events = _events(
                task_id,
                units,
                detections,
                recognitions,
                center_seconds=center,
                max_events=max_events,
                prefer_center=mode == "local",
            )
            task_results.append(
                {
                    "task_id": task_id,
                    "mode": mode,
                    "unit_count": len(units),
                    "recognition_frame_count": len(shortlist),
                    "scan_order": "middle_out",
                    "selected_sample_seconds": [
                        round(units[index].sample_seconds, 3)
                        for index in shortlist
                    ],
                    "positive_sample_seconds": [
                        round(units[index].sample_seconds, 3)
                        for index in shortlist
                        if recognitions[index].positive
                    ],
                    "units": _unit_diagnostics(
                        units,
                        detections,
                        recognitions,
                        shortlist,
                    ),
                    "events": events,
                }
            )
    finally:
        reader.close()
    return {
        "detector_model": DETECTION_MODEL,
        "recognizer_model": RECOGNITION_MODEL,
        "recognizer_loaded": recognizer is not None,
        "recognition_frame_count": total_recognized,
        "tasks": task_results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.request.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise OCRWorkerError("Request must be a JSON object.")
        result = run(payload)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return 0
    except (OCRWorkerError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"OCR worker failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
