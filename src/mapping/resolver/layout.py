from __future__ import annotations

import numpy as np
import pandas as pd

from .common import (detections_without_substitute_panel, is_table_name_like,
                     normalize_text, parse_inline_player, shirt_number_rows,
                     similarity)


FORMATION_LABEL_MIN_GAP = 0.018
FORMATION_LABEL_MAX_GAP = 0.13


def is_large_heading_prefix(row: pd.Series, frame: pd.DataFrame) -> bool:
    """Detect logo text repeated as the prefix of a much larger team heading."""
    required = {"x1", "x2", "y1", "y2"}
    if not required.issubset(frame.columns):
        return False
    label = normalize_text(row["text"])
    if not label:
        return False
    width = max(1.0, float(row["x2"]) - float(row["x1"]))
    height = max(1.0, float(row["y2"]) - float(row["y1"]))
    for _, other in frame.iterrows():
        heading = normalize_text(other["text"])
        if heading == label or not heading.startswith(f"{label} "):
            continue
        heading_width = float(other["x2"]) - float(other["x1"])
        heading_height = float(other["y2"]) - float(other["y1"])
        if heading_width >= 2 * width and heading_height >= 1.8 * height:
            return True
    return False


def formation_anchor_pairs(frame: pd.DataFrame) -> list[tuple[pd.Series, pd.Series]]:
    numbers = shirt_number_rows(frame)
    names = frame[
        (frame["text_type"] != "shirt_number_candidate")
        & frame["text"].map(is_table_name_like)
        & ~frame["text"].map(lambda value: parse_inline_player(value) is not None)
    ]
    anchors: list[tuple[pd.Series, pd.Series]] = []
    used_names: set[int] = set()
    for _, number_row in numbers.iterrows():
        candidates: list[tuple[float, int, pd.Series]] = []
        for name_index, name_row in names.iterrows():
            if int(name_index) in used_names:
                continue
            dx = abs(float(name_row["center_x_norm"]) - float(number_row["center_x_norm"]))
            dy = float(name_row["center_y_norm"]) - float(number_row["center_y_norm"])
            if dx <= 0.065 and FORMATION_LABEL_MIN_GAP <= dy <= FORMATION_LABEL_MAX_GAP:
                candidates.append((dx + abs(dy - 0.06), int(name_index), name_row))
        if candidates:
            _, name_index, name_row = min(candidates, key=lambda item: item[0])
            used_names.add(name_index)
            anchors.append((number_row, name_row))
    return anchors


def formation_refinement_frames(
    segment: pd.DataFrame,
) -> list[tuple[int, list[tuple[pd.Series, pd.Series]]]]:
    formation_segment = detections_without_substitute_panel(segment)
    candidates: list[tuple[int, list[tuple[pd.Series, pd.Series]]]] = []
    for frame_index, frame in formation_segment.groupby("frame_index", sort=True):
        anchors = formation_anchor_pairs(frame)
        if len(anchors) >= 3:
            candidates.append((int(frame_index), anchors))
    if not candidates:
        return []
    maximum_anchors = max(len(item[1]) for item in candidates)
    minimum_anchors = max(3, maximum_anchors - 2)
    best = [item for item in candidates if len(item[1]) >= minimum_anchors]
    best.sort(key=lambda item: item[0])
    positions = [0, (len(best) - 1) // 2, len(best) - 1]
    return list({best[position][0]: best[position] for position in positions}.values())


def recovery_formation_name_rows(frame: pd.DataFrame) -> list[pd.Series]:
    """Return name candidates without deriving a vertical band from detected numbers."""
    candidates: list[pd.Series] = []
    for _, row in frame.iterrows():
        if (
            row["text_type"] == "shirt_number_candidate"
            or not is_table_name_like(row["text"])
            or parse_inline_player(row["text"]) is not None
            or is_large_heading_prefix(row, frame)
        ):
            continue
        candidates.append(row)
    return candidates


def _name_signature_similarity(left: list[pd.Series], right: list[pd.Series]) -> float:
    if not left or not right:
        return 0.0
    used: set[int] = set()
    matches = 0
    for left_row in left:
        best: tuple[float, int] | None = None
        for index, right_row in enumerate(right):
            if index in used or similarity(left_row["text"], right_row["text"]) < 0.82:
                continue
            dx = abs(float(left_row["center_x_norm"]) - float(right_row["center_x_norm"]))
            dy = abs(float(left_row["center_y_norm"]) - float(right_row["center_y_norm"]))
            if dx <= 0.04 and dy <= 0.04:
                distance = dx + dy
                if best is None or distance < best[0]:
                    best = distance, index
        if best is not None:
            used.add(best[1])
            matches += 1
    return matches / max(len(left), len(right))


def formation_recovery_windows(
    segment: pd.DataFrame, expected_players: int, window_size: int = 3,
    maximum_windows: int = 4,
) -> list[list[tuple[int, list[tuple[pd.Series, pd.Series]]]]]:
    """Rank stable name-rich windows used only after normal resolution has failed."""
    formation_segment = detections_without_substitute_panel(segment)
    frames: list[tuple[int, float, list[pd.Series], list[tuple[pd.Series, pd.Series]]]] = []
    for frame_index, frame in formation_segment.groupby("frame_index", sort=True):
        names = recovery_formation_name_rows(frame)
        # A small allowance covers a league logo or team heading. Larger text-heavy
        # screens are too likely to be a sponsor animation or a roster list.
        if expected_players <= len(names) <= expected_players + 3:
            frames.append((
                int(frame_index), float(frame["timestamp_seconds"].iloc[0]), names,
                formation_anchor_pairs(frame),
            ))
    ranked: list[
        tuple[tuple[float, float, float, float], list[tuple[int, list[tuple[pd.Series, pd.Series]]]]]
    ] = []
    for start in range(max(0, len(frames) - window_size + 1)):
        window = frames[start : start + window_size]
        if len(window) != window_size:
            continue
        gaps = [right[1] - left[1] for left, right in zip(window, window[1:])]
        if any(gap <= 0 or gap > 0.75 for gap in gaps):
            continue
        stability = float(np.mean([
            _name_signature_similarity(left[2], right[2])
            for left, right in zip(window, window[1:])
        ]))
        if stability < 0.60:
            continue
        mean_anchors = float(np.mean([len(item[3]) for item in window]))
        mean_names = float(np.mean([len(item[2]) for item in window]))
        score = (
            mean_anchors,
            stability,
            -abs(mean_names - (expected_players + 1)),
            window[-1][1],
        )
        ranked.append((score, [(item[0], item[3]) for item in window]))
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected: list[list[tuple[int, list[tuple[pd.Series, pd.Series]]]]] = []
    occupied: set[int] = set()
    for _, window in ranked:
        indices = {item[0] for item in window}
        if indices & occupied:
            continue
        selected.append(window)
        occupied.update(indices)
        if len(selected) == maximum_windows:
            break
    return selected


def formation_name_rows(
    frame: pd.DataFrame, anchors: list[tuple[pd.Series, pd.Series]]
) -> list[pd.Series]:
    if not anchors:
        return []
    anchor_names = [name_row for _, name_row in anchors]
    y_values = [float(name_row["center_y_norm"]) for name_row in anchor_names]
    minimum_y = max(0.0, min(y_values) - 0.20)
    maximum_y = min(1.0, max(y_values) + 0.35)
    candidates: list[pd.Series] = []
    for _, row in frame.iterrows():
        if (
            row["text_type"] == "shirt_number_candidate"
            or not is_table_name_like(row["text"])
            or parse_inline_player(row["text"]) is not None
            or is_large_heading_prefix(row, frame)
        ):
            continue
        y = float(row["center_y_norm"])
        if minimum_y <= y <= maximum_y:
            candidates.append(row)
    return candidates


def formation_number_gap(anchors: list[tuple[pd.Series, pd.Series]]) -> float:
    gaps = [
        float(name_row["center_y_norm"]) - float(number_row["center_y_norm"])
        for number_row, name_row in anchors
    ]
    if not gaps:
        return 0.05
    return float(np.clip(np.median(gaps), 0.03, 0.13))
