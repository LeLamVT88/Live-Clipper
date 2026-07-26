"""Targeted OCR for shirt numbers positioned above formation names."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    LineupResolutionError,
    detections_without_substitute_panel,
    resolve_project_path,
)
from .layout import (
    formation_name_rows,
    formation_number_gap,
    formation_refinement_frames,
)
from .local_models import (
    number_preprocessing_variants,
    numeric_candidate_consensus,
    numeric_detection,
    recognition_numeric_detection,
    resized_number_crop,
)


def refine_formation_numbers(
    segment: pd.DataFrame,
    ocr: object,
    recognizer: object | None = None,
) -> tuple[pd.DataFrame, int]:
    formation_segment = detections_without_substitute_panel(segment)
    chosen = formation_refinement_frames(formation_segment)
    if not chosen or "frame_path" not in segment.columns:
        return segment, 0

    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise LineupResolutionError(
            "OpenCV is required for local formation OCR."
        ) from exc

    added_rows: list[dict[str, object]] = []
    for frame_index, anchors in chosen:
        frame = formation_segment[
            formation_segment["frame_index"] == frame_index
        ]
        frame_path = resolve_project_path(
            Path(str(frame["frame_path"].iloc[0]))
        )
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        height, width = image.shape[:2]

        name_rows = formation_name_rows(frame, anchors)
        if not name_rows:
            continue
        number_gap = formation_number_gap(anchors)

        valid_rows = list(name_rows)
        crops = [
            resized_number_crop(
                image,
                center_x=float(row["center_x_norm"]),
                center_y=(
                    float(row["center_y_norm"])
                    - number_gap
                ),
                x_radius=0.020,
                y_radius=0.030,
                scale=8,
                round_coordinates=True,
            )
            for row in valid_rows
        ]
        detector_results = list(ocr.predict(crops))
        if len(detector_results) != len(valid_rows):
            continue
        candidates_by_row: list[
            list[tuple[int, float] | None]
        ] = [
            [numeric_detection(result)]
            for result in detector_results
        ]

        missing_indices = [
            index
            for index, candidates in enumerate(candidates_by_row)
            if candidates[0] is None
        ]
        if missing_indices:
            fallback_crops = [
                resized_number_crop(
                    image,
                    center_x=float(
                        valid_rows[index]["center_x_norm"]
                    ),
                    center_y=(
                        float(
                            valid_rows[index][
                                "center_y_norm"
                            ]
                        )
                        - number_gap
                    ),
                    x_radius=0.022,
                    y_radius=0.025,
                    scale=7,
                )
                for index in missing_indices
            ]
            fallback_results = list(
                ocr.predict(fallback_crops)
            )
            for index, result in zip(
                missing_indices,
                fallback_results,
                strict=True,
            ):
                candidates_by_row[index].append(
                    numeric_detection(result)
                )

        if recognizer is not None:
            recognition_crops: list[np.ndarray] = []
            recognition_owners: list[int] = []
            for index, crop in enumerate(crops):
                variants = number_preprocessing_variants(crop)
                recognition_crops.extend(variants)
                recognition_owners.extend(
                    [index] * len(variants)
                )
            recognition_results = list(
                recognizer.predict(recognition_crops)
            )
            if len(recognition_results) != len(
                recognition_owners
            ):
                raise LineupResolutionError(
                    "Local number recognizer returned an "
                    "unexpected result count."
                )
            for index, result in zip(
                recognition_owners,
                recognition_results,
                strict=True,
            ):
                candidates_by_row[index].append(
                    recognition_numeric_detection(result)
                )

        numeric_results = [
            numeric_candidate_consensus(candidates)
            for candidates in candidates_by_row
        ]

        for name_row, numeric in zip(
            valid_rows,
            numeric_results,
            strict=True,
        ):
            if numeric is None:
                continue
            shirt_number, score = numeric
            center_x = float(name_row["center_x_norm"])
            center_y = (
                float(name_row["center_y_norm"])
                - number_gap
            )
            output_row = name_row.to_dict()
            output_row.update(
                {
                    "text": str(shirt_number),
                    "text_type": "shirt_number_candidate",
                    "score": round(score, 6),
                    "center_x_norm": round(center_x, 6),
                    "center_y_norm": round(center_y, 6),
                }
            )
            if {
                "x1",
                "x2",
                "y1",
                "y2",
                "center_x",
                "center_y",
            }.issubset(segment.columns):
                center_x_pixels = center_x * width
                center_y_pixels = center_y * height
                output_row.update(
                    {
                        "x1": int(
                            round(center_x_pixels - 18)
                        ),
                        "x2": int(
                            round(center_x_pixels + 18)
                        ),
                        "y1": int(
                            round(center_y_pixels - 18)
                        ),
                        "y2": int(
                            round(center_y_pixels + 18)
                        ),
                        "center_x": round(
                            center_x_pixels,
                            3,
                        ),
                        "center_y": round(
                            center_y_pixels,
                            3,
                        ),
                    }
                )
            added_rows.append(output_row)

    if not added_rows:
        return segment, 0
    refined = pd.concat(
        [
            segment,
            pd.DataFrame(
                added_rows,
                columns=segment.columns,
            ),
        ],
        ignore_index=True,
    )
    return refined, len(added_rows)
