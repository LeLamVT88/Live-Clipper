"""Reconstruct full player names from nearby OCR labels."""

from __future__ import annotations

import re
from itertools import combinations

import numpy as np
import pandas as pd

from .common import (
    consensus_text,
    is_name_like,
    normalize_text,
    similarity,
)


NameCandidate = tuple[str, float, int]


def label_occurrences(
    formation_label: str,
    detections: pd.DataFrame,
) -> pd.DataFrame:
    return detections[
        detections.apply(
            lambda row: (
                row["text_type"] != "shirt_number_candidate"
                and similarity(row["text"], formation_label) >= 0.82
            ),
            axis=1,
        )
    ]


def nearby_name_candidates(
    formation_label: str,
    occurrences: pd.DataFrame,
    detections: pd.DataFrame,
    other_labels: set[str],
) -> list[NameCandidate]:
    candidates: list[NameCandidate] = []
    label_normalized = normalize_text(formation_label)

    for _, label_row in occurrences.iterrows():
        frame_index = int(label_row["frame_index"])
        frame = detections[
            (detections["frame_index"] == frame_index)
            & (detections["text_type"] != "shirt_number_candidate")
        ]
        neighbors: list[tuple[float, pd.Series]] = []
        for _, row in frame.iterrows():
            normalized = normalize_text(row["text"])
            if normalized == normalize_text(label_row["text"]):
                continue
            if (
                not is_name_like(row["text"])
                or normalized in other_labels
            ):
                continue
            dx = abs(
                float(row["center_x_norm"])
                - float(label_row["center_x_norm"])
            )
            dy = abs(
                float(row["center_y_norm"])
                - float(label_row["center_y_norm"])
            )
            if dx <= 0.085 and 0.012 <= dy <= 0.09:
                neighbors.append((dy + 0.5 * dx, row))

        if not neighbors:
            continue
        neighbor = min(neighbors, key=lambda item: item[0])[1]
        ordered = sorted(
            [label_row, neighbor],
            key=lambda row: (
                float(row["center_y_norm"]),
                float(row["center_x_norm"]),
            ),
        )
        combined = " ".join(
            str(row["text"]).strip() for row in ordered
        )
        combined_normalized = normalize_text(combined)
        if (
            label_normalized not in combined_normalized
            or len(combined_normalized.split()) > 6
        ):
            continue
        candidates.append(
            (
                combined,
                float(label_row["score"]) * float(neighbor["score"]),
                frame_index,
            )
        )
    return candidates


def multiline_name_candidates(
    formation_label: str,
    detections: pd.DataFrame,
    other_labels: set[str],
) -> list[NameCandidate]:
    candidates: list[NameCandidate] = []
    label_normalized = normalize_text(formation_label)

    for frame_index, frame in detections.groupby(
        "frame_index",
        sort=False,
    ):
        name_rows = [
            row
            for _, row in frame.iterrows()
            if (
                row["text_type"] != "shirt_number_candidate"
                and is_name_like(row["text"])
                and (
                    normalize_text(row["text"]) not in other_labels
                    or similarity(row["text"], formation_label) >= 0.82
                )
            )
        ]
        for group_size in (2, 3):
            for rows in combinations(name_rows, group_size):
                if not rows_form_name(rows):
                    continue
                ordered = sorted(
                    rows,
                    key=lambda row: float(row["center_y_norm"]),
                )
                combined = re.sub(
                    r"\s*-\s*",
                    "-",
                    " ".join(
                        str(row["text"]).strip() for row in ordered
                    ),
                )
                combined_normalized = normalize_text(combined)
                if (
                    label_normalized not in combined_normalized
                    or len(combined_normalized) <= len(label_normalized)
                ):
                    continue
                candidates.append(
                    (
                        combined,
                        float(
                            np.mean(
                                [float(row["score"]) for row in ordered]
                            )
                        ),
                        int(frame_index),
                    )
                )
    return candidates


def rows_form_name(rows: tuple[pd.Series, ...]) -> bool:
    x_values = [float(row["center_x_norm"]) for row in rows]
    if max(x_values) - min(x_values) > 0.065:
        return False

    y_values = sorted(float(row["center_y_norm"]) for row in rows)
    return (
        y_values[-1] - y_values[0] <= 0.14
        and all(
            right - left <= 0.09
            for left, right in zip(y_values, y_values[1:])
        )
    )


def deduplicate_candidates(
    candidates: list[NameCandidate],
) -> list[NameCandidate]:
    deduplicated: dict[tuple[int, str], NameCandidate] = {}
    for candidate in candidates:
        key = (candidate[2], normalize_text(candidate[0]))
        existing = deduplicated.get(key)
        if existing is None or candidate[1] > existing[1]:
            deduplicated[key] = candidate
    return list(deduplicated.values())


def best_full_name(
    formation_label: str,
    event_detections: pd.DataFrame,
    other_formation_labels: set[str],
) -> tuple[str, float, int]:
    occurrences = label_occurrences(
        formation_label,
        event_detections,
    )
    candidates = deduplicate_candidates(
        nearby_name_candidates(
            formation_label,
            occurrences,
            event_detections,
            other_formation_labels,
        )
        + multiline_name_candidates(
            formation_label,
            event_detections,
            other_formation_labels,
        )
    )

    if not candidates:
        confidence = (
            float(occurrences["score"].mean())
            if not occurrences.empty
            else 0.0
        )
        evidence = int(occurrences["frame_index"].nunique())
        return formation_label, confidence, evidence

    full_name, confidence, evidence = consensus_text(
        candidates,
        fuzzy_threshold=0.87,
    )
    if (
        len(normalize_text(full_name))
        <= len(normalize_text(formation_label))
    ):
        return formation_label, confidence, evidence
    return full_name, confidence, evidence
