from __future__ import annotations

import numpy as np
import pandas as pd

from .common import LineupResolutionError, detections_without_substitute_panel
from .layout import (
    formation_name_rows,
    formation_number_gap,
    formation_refinement_frames,
)
from .local_models import (
    PIXEL_GEOMETRY_COLUMNS,
    append_refinement_rows,
    load_frame_image,
    number_preprocessing_variants,
    numeric_candidate_consensus,
    numeric_detection,
    numeric_observation,
    recognition_numeric_detection,
    require_cv2,
    resized_number_crop,
)


def refine_formation_numbers(
    segment: pd.DataFrame, ocr: object, recognizer: object | None = None,
) -> tuple[pd.DataFrame, int]:
    formation_segment = detections_without_substitute_panel(segment)
    chosen = formation_refinement_frames(formation_segment)
    if not chosen or "frame_path" not in segment.columns:
        return segment, 0
    cv2 = require_cv2("formation")
    include_pixels = PIXEL_GEOMETRY_COLUMNS.issubset(segment.columns)
    added_rows: list[dict[str, object]] = []
    for frame_index, anchors in chosen:
        frame = formation_segment[formation_segment["frame_index"] == frame_index]
        image = load_frame_image(frame["frame_path"].iloc[0], cv2)
        if image is None:
            continue
        height, width = image.shape[:2]
        name_rows = formation_name_rows(frame, anchors)
        if not name_rows:
            continue
        number_gap = formation_number_gap(anchors)
        crops = [
            resized_number_crop(
                image, float(row["center_x_norm"]),
                float(row["center_y_norm"]) - number_gap, 0.020, 0.030, 8, True,
            )
            for row in name_rows
        ]
        detector_results = list(ocr.predict(crops))
        if len(detector_results) != len(name_rows):
            continue
        candidates_by_row: list[list[tuple[int, float] | None]] = [
            [numeric_detection(result)] for result in detector_results
        ]
        missing_indices = [
            index for index, candidates in enumerate(candidates_by_row) if candidates[0] is None
        ]
        if missing_indices:
            fallback_crops = [
                resized_number_crop(
                    image, float(name_rows[index]["center_x_norm"]),
                    float(name_rows[index]["center_y_norm"]) - number_gap, 0.022, 0.025, 7,
                )
                for index in missing_indices
            ]
            fallback_results = list(ocr.predict(fallback_crops))
            for index, result in zip(missing_indices, fallback_results, strict=True):
                candidates_by_row[index].append(numeric_detection(result))
        if recognizer is not None:
            recognition_crops: list[np.ndarray] = []
            recognition_owners: list[int] = []
            for index, crop in enumerate(crops):
                variants = number_preprocessing_variants(crop)
                recognition_crops.extend(variants)
                recognition_owners.extend([index] * len(variants))
            recognition_results = list(recognizer.predict(recognition_crops))
            if len(recognition_results) != len(recognition_owners):
                raise LineupResolutionError("Local number recognizer returned an unexpected result count.")
            for index, result in zip(recognition_owners, recognition_results, strict=True):
                candidates_by_row[index].append(recognition_numeric_detection(result))
        for name_row, candidates in zip(name_rows, candidates_by_row, strict=True):
            numeric = numeric_candidate_consensus(candidates)
            if numeric is None:
                continue
            shirt_number, score = numeric
            center_x = float(name_row["center_x_norm"])
            center_y = float(name_row["center_y_norm"]) - number_gap
            added_rows.append(
                numeric_observation(
                    name_row,
                    shirt_number=shirt_number,
                    score=score,
                    center_x=center_x,
                    center_y=center_y,
                    image_width=width,
                    image_height=height,
                    half_width=18,
                    half_height=18,
                    include_pixel_geometry=include_pixels,
                )
            )
    return append_refinement_rows(segment, added_rows)
