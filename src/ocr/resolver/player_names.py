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


def best_full_name(
    formation_label: str,
    event_detections: pd.DataFrame,
    other_formation_labels: set[str],
) -> tuple[str, float, int]:
    candidates: list[tuple[str, float, int]] = []
    label_normalized = normalize_text(formation_label)
    label_occurrences = event_detections[
        event_detections.apply(
            lambda row: (
                row["text_type"]
                != "shirt_number_candidate"
                and similarity(
                    row["text"],
                    formation_label,
                )
                >= 0.82
            ),
            axis=1,
        )
    ]

    for _, label_row in label_occurrences.iterrows():
        frame_index = int(label_row["frame_index"])
        frame = event_detections[
            (
                event_detections["frame_index"]
                == frame_index
            )
            & (
                event_detections["text_type"]
                != "shirt_number_candidate"
            )
        ]
        neighbors: list[tuple[float, pd.Series]] = []
        for _, row in frame.iterrows():
            text = str(row["text"])
            normalized = normalize_text(text)
            if normalized == normalize_text(label_row["text"]):
                continue
            if (
                not is_name_like(text)
                or normalized in other_formation_labels
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
        neighbor = min(
            neighbors,
            key=lambda item: item[0],
        )[1]
        ordered = sorted(
            [label_row, neighbor],
            key=lambda row: (
                float(row["center_y_norm"]),
                float(row["center_x_norm"]),
            ),
        )
        combined = " ".join(
            str(row["text"]).strip()
            for row in ordered
        )
        combined_normalized = normalize_text(combined)
        if label_normalized not in combined_normalized:
            continue
        if len(combined_normalized.split()) > 6:
            continue
        candidates.append(
            (
                combined,
                float(label_row["score"])
                * float(neighbor["score"]),
                frame_index,
            )
        )

    for frame_index, frame in event_detections.groupby(
        "frame_index",
        sort=False,
    ):
        name_rows = [
            row
            for _, row in frame.iterrows()
            if (
                row["text_type"]
                != "shirt_number_candidate"
                and is_name_like(row["text"])
                and (
                    normalize_text(row["text"])
                    not in other_formation_labels
                    or similarity(
                        row["text"],
                        formation_label,
                    )
                    >= 0.82
                )
            )
        ]
        for group_size in (2, 3):
            for rows in combinations(
                name_rows,
                group_size,
            ):
                x_values = [
                    float(row["center_x_norm"])
                    for row in rows
                ]
                y_values = [
                    float(row["center_y_norm"])
                    for row in rows
                ]
                if max(x_values) - min(x_values) > 0.065:
                    continue
                ordered = sorted(
                    rows,
                    key=lambda row: float(
                        row["center_y_norm"]
                    ),
                )
                ordered_y = [
                    float(row["center_y_norm"])
                    for row in ordered
                ]
                if ordered_y[-1] - ordered_y[0] > 0.14:
                    continue
                if any(
                    right - left > 0.09
                    for left, right in zip(
                        ordered_y,
                        ordered_y[1:],
                    )
                ):
                    continue

                combined = " ".join(
                    str(row["text"]).strip()
                    for row in ordered
                )
                combined = re.sub(
                    r"\s*-\s*",
                    "-",
                    combined,
                )
                combined_normalized = normalize_text(combined)
                if label_normalized not in combined_normalized:
                    continue
                if (
                    len(combined_normalized)
                    <= len(label_normalized)
                ):
                    continue
                candidates.append(
                    (
                        combined,
                        float(
                            np.mean(
                                [
                                    float(row["score"])
                                    for row in ordered
                                ]
                            )
                        ),
                        int(frame_index),
                    )
                )

    deduplicated: dict[
        tuple[int, str],
        tuple[str, float, int],
    ] = {}
    for candidate in candidates:
        key = (
            candidate[2],
            normalize_text(candidate[0]),
        )
        existing = deduplicated.get(key)
        if existing is None or candidate[1] > existing[1]:
            deduplicated[key] = candidate
    candidates = list(deduplicated.values())

    if not candidates:
        fallback = label_occurrences
        confidence = (
            float(fallback["score"].mean())
            if not fallback.empty
            else 0.0
        )
        evidence = int(
            fallback["frame_index"].nunique()
        )
        return formation_label, confidence, evidence

    full_name, confidence, evidence = consensus_text(
        candidates,
        fuzzy_threshold=0.87,
    )
    if (
        len(normalize_text(full_name))
        <= len(label_normalized)
    ):
        return formation_label, confidence, evidence
    return full_name, confidence, evidence
