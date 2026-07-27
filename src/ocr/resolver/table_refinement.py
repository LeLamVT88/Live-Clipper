"""Targeted OCR for missing shirt numbers in vertical lineup tables."""

from __future__ import annotations

from collections import defaultdict
import numpy as np
import pandas as pd

from .common import (
    PairObservation,
)
from .local_models import (
    PIXEL_GEOMETRY_COLUMNS,
    append_refinement_rows,
    load_frame_image,
    local_result_data,
    numeric_observation,
    require_cv2,
)
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

    cv2 = require_cv2("table")
    include_pixels = PIXEL_GEOMETRY_COLUMNS.issubset(segment.columns)

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

        image = load_frame_image(
            frame["frame_path"].iloc[0],
            cv2,
        )
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
            center_y = float(name_row["center_y_norm"])
            added_rows.append(
                numeric_observation(
                    name_row,
                    shirt_number=shirt_number,
                    score=score,
                    center_x=number_center_x,
                    center_y=center_y,
                    image_width=width,
                    image_height=height,
                    half_width=20,
                    half_height=half_height,
                    include_pixel_geometry=include_pixels,
                )
            )

    return append_refinement_rows(segment, added_rows)
