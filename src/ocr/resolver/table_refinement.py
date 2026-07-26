"""Targeted OCR for missing shirt numbers in vertical lineup tables."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    LineupResolutionError,
    PairObservation,
    resolve_project_path,
)
from .local_models import local_result_data
from .table import table_pair_observations, table_rows_for_frame


def refine_table_numbers(
    segment: pd.DataFrame,
    ocr: object,
    expected_players: int,
) -> tuple[pd.DataFrame, int]:
    if "frame_path" not in segment.columns:
        return segment, 0

    observations = [
        observation
        for observation in table_pair_observations(segment)
        if observation.relation == "right"
    ]
    by_frame: defaultdict[
        int,
        list[PairObservation],
    ] = defaultdict(list)
    for observation in observations:
        by_frame[observation.frame_index].append(observation)
    usable_frames = [
        (frame_index, frame_observations)
        for frame_index, frame_observations in by_frame.items()
        if len(frame_observations) >= 2
    ]
    if not usable_frames:
        return segment, 0

    earliest = min(usable_frames, key=lambda item: item[0])
    ranked = sorted(
        usable_frames,
        key=lambda item: (len(item[1]), -item[0]),
        reverse=True,
    )
    chosen: list[tuple[int, list[PairObservation]]] = [earliest]
    for candidate in ranked:
        if candidate[0] != earliest[0]:
            chosen.append(candidate)
        if len(chosen) == 3:
            break

    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise LineupResolutionError(
            "OpenCV is required for local table OCR."
        ) from exc

    added_rows: list[dict[str, object]] = []
    for frame_index, frame_observations in chosen:
        frame = segment[segment["frame_index"] == frame_index]
        rows = table_rows_for_frame(
            frame,
            frame_observations,
            expected_players=expected_players,
        )
        if len(rows) != expected_players:
            continue

        frame_path = resolve_project_path(
            Path(str(frame["frame_path"].iloc[0]))
        )
        if not frame_path.is_file():
            continue
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        number_center_x = float(
            np.median(
                [
                    observation.center_x
                    for observation in frame_observations
                ]
            )
        )
        y_values = [
            float(row["center_y_norm"])
            for row in rows
        ]
        row_gap = float(np.median(np.diff(y_values)))
        x1 = max(
            0,
            int(round((number_center_x - 0.035) * width)),
        )
        x2 = min(
            width,
            int(round((number_center_x + 0.035) * width)),
        )
        half_height = max(
            18,
            int(round(row_gap * height * 0.43)),
        )

        crops: list[np.ndarray] = []
        for row in rows:
            center_y = int(
                round(float(row["center_y_norm"]) * height)
            )
            crop = image[
                max(0, center_y - half_height) : min(
                    height,
                    center_y + half_height,
                ),
                x1:x2,
            ]
            crops.append(
                cv2.resize(
                    crop,
                    None,
                    fx=3,
                    fy=3,
                    interpolation=cv2.INTER_CUBIC,
                )
            )

        results = list(ocr.predict(crops))
        if len(results) != len(rows):
            continue
        for name_row, result in zip(rows, results, strict=True):
            data = local_result_data(result)
            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])
            numeric = [
                (int(str(text).strip()), float(score))
                for text, score in zip(
                    texts,
                    scores,
                    strict=False,
                )
                if str(text).strip().isdigit()
                and 1 <= int(str(text).strip()) <= 99
                and float(score) >= 0.60
            ]
            if not numeric:
                continue
            shirt_number, score = max(
                numeric,
                key=lambda item: item[1],
            )
            output_row = name_row.to_dict()
            center_y = float(name_row["center_y_norm"])
            output_row.update(
                {
                    "text": str(shirt_number),
                    "text_type": "shirt_number_candidate",
                    "score": round(score, 6),
                    "center_x_norm": round(
                        number_center_x,
                        6,
                    ),
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
                center_x_pixels = number_center_x * width
                center_y_pixels = center_y * height
                output_row.update(
                    {
                        "x1": int(
                            round(center_x_pixels - 20)
                        ),
                        "x2": int(
                            round(center_x_pixels + 20)
                        ),
                        "y1": int(
                            round(
                                center_y_pixels - half_height
                            )
                        ),
                        "y2": int(
                            round(
                                center_y_pixels + half_height
                            )
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
