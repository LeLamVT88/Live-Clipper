from __future__ import annotations

import re
from itertools import combinations

import numpy as np
import pandas as pd

from .common import (
    consensus_text,
    is_name_like,
    normalize_text,
    parse_inline_player,
    similarity,
)
from .table import table_pair_observations


NameCandidate = tuple[str, float, int]


def label_occurrences(formation_label: str, detections: pd.DataFrame) -> pd.DataFrame:
    return detections[
        detections.apply(
            lambda row: row["text_type"] != "shirt_number_candidate"
            and similarity(row["text"], formation_label) >= 0.82,
            axis=1,
        )
    ]


def nearby_name_candidates(
    formation_label: str, occurrences: pd.DataFrame,
    detections: pd.DataFrame, other_labels: set[str],
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
                or parse_inline_player(row["text"]) is not None
                or normalized in other_labels
            ):
                continue
            dx = abs(float(row["center_x_norm"]) - float(label_row["center_x_norm"]))
            dy = abs(float(row["center_y_norm"]) - float(label_row["center_y_norm"]))
            if dx <= 0.085 and 0.012 <= dy <= 0.09:
                neighbors.append((dy + 0.5 * dx, row))
        if not neighbors:
            continue
        neighbor = min(neighbors, key=lambda item: item[0])[1]
        ordered = sorted(
            [label_row, neighbor],
            key=lambda row: (float(row["center_y_norm"]), float(row["center_x_norm"])),
        )
        combined = " ".join(str(row["text"]).strip() for row in ordered)
        combined_normalized = normalize_text(combined)
        if label_normalized not in combined_normalized or len(combined_normalized.split()) > 6:
            continue
        candidates.append((combined, float(label_row["score"]) * float(neighbor["score"]), frame_index))
    return candidates


def multiline_name_candidates(
    formation_label: str, detections: pd.DataFrame, other_labels: set[str],
) -> list[NameCandidate]:
    candidates: list[NameCandidate] = []
    label_normalized = normalize_text(formation_label)
    for frame_index, frame in detections.groupby("frame_index", sort=False):
        name_rows = [
            row
            for _, row in frame.iterrows()
            if (
                row["text_type"] != "shirt_number_candidate"
                and is_name_like(row["text"])
                and parse_inline_player(row["text"]) is None
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
                ordered = sorted(rows, key=lambda row: float(row["center_y_norm"]))
                combined = re.sub(
                    r"\s*-\s*",
                    "-",
                    " ".join(str(row["text"]).strip() for row in ordered),
                )
                combined_normalized = normalize_text(combined)
                if label_normalized not in combined_normalized or len(combined_normalized) <= len(
                    label_normalized
                ):
                    continue
                candidates.append((
                    combined, float(np.mean([float(row["score"]) for row in ordered])),
                    int(frame_index),
                ))
    return candidates


def rows_form_name(rows: tuple[pd.Series, ...]) -> bool:
    x_values = [float(row["center_x_norm"]) for row in rows]
    if max(x_values) - min(x_values) > 0.065:
        return False
    y_values = sorted(float(row["center_y_norm"]) for row in rows)
    return y_values[-1] - y_values[0] <= 0.14 and all(
        right - left <= 0.09 for left, right in zip(y_values, y_values[1:])
    )


def deduplicate_candidates(candidates: list[NameCandidate]) -> list[NameCandidate]:
    deduplicated: dict[tuple[int, str], NameCandidate] = {}
    for candidate in candidates:
        key = (candidate[2], normalize_text(candidate[0]))
        existing = deduplicated.get(key)
        if existing is None or candidate[1] > existing[1]:
            deduplicated[key] = candidate
    return list(deduplicated.values())


def best_full_name(
    formation_label: str, event_detections: pd.DataFrame,
    other_formation_labels: set[str],
) -> tuple[str, float, int]:
    occurrences = label_occurrences(formation_label, event_detections)
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
        confidence = float(occurrences["score"].mean()) if not occurrences.empty else 0.0
        evidence = int(occurrences["frame_index"].nunique())
        return formation_label, confidence, evidence
    full_name, confidence, evidence = consensus_text(candidates, fuzzy_threshold=0.87)
    if len(normalize_text(full_name)) <= len(normalize_text(formation_label)):
        return formation_label, confidence, evidence
    return full_name, confidence, evidence


def _is_name_reference(
    current_name: str, candidate: str, *, shirt_linked: bool = False,
) -> bool:
    current = normalize_text(current_name)
    reference = normalize_text(candidate)
    if not current or not reference:
        return False
    if reference == current:
        return True
    if len(reference) <= len(current):
        return False
    current_tokens = current.split()
    reference_tokens = reference.split()
    added_tokens = reference_tokens[: max(0, len(reference_tokens) - len(current_tokens))]
    if shirt_linked and any(len(token) > 1 for token in added_tokens):
        return True
    reference_last = reference_tokens[-1]
    current_last = current_tokens[-1]
    matches = (
        similarity(current, reference) >= 0.82
        or similarity(current, reference_last) >= 0.86
        or similarity(current_last, reference_last) >= 0.90
    )
    if not matches:
        return False
    return not added_tokens or any(len(token) > 1 for token in added_tokens)


def enrich_player_names(
    records: list[dict[str, object]], reference_detections: pd.DataFrame,
    minimum_evidence_frames: int = 2,
) -> list[dict[str, object]]:
    """Use sparse full-frame OCR only to expand names already resolved in a lineup."""
    if not records or reference_detections.empty:
        return records
    references = reference_detections[
        reference_detections.apply(
            lambda row: row["text_type"] != "shirt_number_candidate"
            and is_name_like(row["text"])
            and parse_inline_player(row["text"]) is None,
            axis=1,
        )
    ]
    paired_references = table_pair_observations(reference_detections)
    enriched: list[dict[str, object]] = []
    for record in records:
        current = str(record.get("player_name", "")).strip()
        try:
            shirt_number = int(record.get("shirt_number"))
        except (TypeError, ValueError):
            shirt_number = -1
        number_linked = [
            (
                item.player_name,
                (item.number_score * item.name_score) ** 0.5,
                item.frame_index,
            )
            for item in paired_references
            if item.shirt_number == shirt_number
        ]
        text_linked = [
            (str(row["text"]).strip(), float(row["score"]), int(row["frame_index"]))
            for _, row in references.iterrows()
            if _is_name_reference(current, str(row["text"]))
        ]
        supported: tuple[str, float, int] | None = None
        for candidates in (number_linked, text_linked):
            if not candidates:
                continue
            candidate = consensus_text(candidates, fuzzy_threshold=0.87)
            is_supported_name = candidates is text_linked or _is_name_reference(
                current, candidate[0], shirt_linked=True
            )
            if candidate[2] >= minimum_evidence_frames and is_supported_name:
                supported = candidate
                break
        if supported is not None:
            name, confidence, evidence = supported
            updated = dict(record)
            updated["player_name"] = name.strip().upper()
            updated["name_source"] = "scout"
            updated["name_confidence"] = round(confidence, 6)
            updated["full_name_evidence_frames"] = evidence
            number_confidence = float(updated.get("number_confidence", 0.0))
            label_confidence = float(updated.get("label_confidence", 0.0))
            updated["pair_confidence"] = round(
                (number_confidence * label_confidence * max(confidence, 0.01)) ** (1 / 3),
                6,
            )
            enriched.append(updated)
            continue
        enriched.append(record)
    return enriched
