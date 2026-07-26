"""Layout anchors shared by scout selection and formation refinement."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import (
    detections_without_substitute_panel,
    is_table_name_like,
    parse_inline_player,
    shirt_number_rows,
)


def formation_anchor_pairs(
    frame: pd.DataFrame,
) -> list[tuple[pd.Series, pd.Series]]:
    numbers = shirt_number_rows(frame)
    names = frame[
        (frame["text_type"] != "shirt_number_candidate")
        & frame["text"].map(is_table_name_like)
        & ~frame["text"].map(
            lambda value: parse_inline_player(value) is not None
        )
    ]
    anchors: list[tuple[pd.Series, pd.Series]] = []
    used_names: set[int] = set()
    for _, number_row in numbers.iterrows():
        candidates: list[tuple[float, int, pd.Series]] = []
        for name_index, name_row in names.iterrows():
            if int(name_index) in used_names:
                continue
            dx = abs(
                float(name_row["center_x_norm"])
                - float(number_row["center_x_norm"])
            )
            dy = (
                float(name_row["center_y_norm"])
                - float(number_row["center_y_norm"])
            )
            if dx <= 0.065 and 0.025 <= dy <= 0.13:
                candidates.append(
                    (
                        dx + abs(dy - 0.06),
                        int(name_index),
                        name_row,
                    )
                )
        if candidates:
            _, name_index, name_row = min(
                candidates,
                key=lambda item: item[0],
            )
            used_names.add(name_index)
            anchors.append((number_row, name_row))
    return anchors


def formation_refinement_frames(
    segment: pd.DataFrame,
) -> list[tuple[int, list[tuple[pd.Series, pd.Series]]]]:
    formation_segment = detections_without_substitute_panel(segment)
    candidates: list[
        tuple[int, list[tuple[pd.Series, pd.Series]]]
    ] = []
    for frame_index, frame in formation_segment.groupby(
        "frame_index",
        sort=True,
    ):
        anchors = formation_anchor_pairs(frame)
        if len(anchors) >= 3:
            candidates.append((int(frame_index), anchors))
    if not candidates:
        return []

    maximum_anchors = max(len(item[1]) for item in candidates)
    minimum_anchors = max(3, maximum_anchors - 2)
    best = [
        item
        for item in candidates
        if len(item[1]) >= minimum_anchors
    ]
    best.sort(key=lambda item: item[0])
    positions = [0, (len(best) - 1) // 2, len(best) - 1]
    chosen: list[
        tuple[int, list[tuple[pd.Series, pd.Series]]]
    ] = []
    for position in positions:
        candidate = best[position]
        if all(
            candidate[0] != existing[0]
            for existing in chosen
        ):
            chosen.append(candidate)
    return chosen


def formation_name_rows(
    frame: pd.DataFrame,
    anchors: list[tuple[pd.Series, pd.Series]],
) -> list[pd.Series]:
    if not anchors:
        return []

    anchor_names = [name_row for _, name_row in anchors]
    y_values = [
        float(name_row["center_y_norm"])
        for name_row in anchor_names
    ]
    minimum_y = max(0.0, min(y_values) - 0.20)
    maximum_y = min(1.0, max(y_values) + 0.35)

    candidates: list[pd.Series] = []
    for _, row in frame.iterrows():
        if (
            row["text_type"] == "shirt_number_candidate"
            or not is_table_name_like(row["text"])
            or parse_inline_player(row["text"]) is not None
        ):
            continue
        y = float(row["center_y_norm"])
        if minimum_y <= y <= maximum_y:
            candidates.append(row)
    return candidates


def formation_number_gap(
    anchors: list[tuple[pd.Series, pd.Series]],
) -> float:
    gaps = [
        float(name_row["center_y_norm"])
        - float(number_row["center_y_norm"])
        for number_row, name_row in anchors
    ]
    return float(np.clip(np.median(gaps), 0.03, 0.13))
