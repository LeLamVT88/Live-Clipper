from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .common import (
    PairObservation,
    consensus_text,
    detections_before_substitutes,
    is_table_name_like,
    normalize_text,
    parse_inline_player,
    shirt_number_rows,
    similarity,
)


def table_pair_observations(segment: pd.DataFrame) -> list[PairObservation]:
    observations: list[PairObservation] = []
    relevant = detections_before_substitutes(segment)
    has_boxes = {"x1", "frame_width"}.issubset(relevant.columns)
    for frame_index, frame in relevant.groupby("frame_index", sort=True):
        timestamp = float(frame["timestamp_seconds"].iloc[0])
        for _, row in frame.iterrows():
            inline = parse_inline_player(row["text"])
            if inline is None:
                continue
            shirt_number, player_name = inline
            label_x1 = (
                float(row["x1"]) / float(row["frame_width"])
                if has_boxes and float(row["frame_width"]) > 0
                else float(row["center_x_norm"])
            )
            observations.append(
                PairObservation(
                    shirt_number, player_name, float(row["score"]), float(row["score"]),
                    int(frame_index), timestamp, float(row["center_x_norm"]),
                    float(row["center_y_norm"]), label_x1, "inline",
                )
            )
        numbers = shirt_number_rows(frame)
        names = frame[
            (frame["text_type"] != "shirt_number_candidate")
            & frame["text"].map(is_table_name_like)
            & ~frame["text"].map(lambda value: parse_inline_player(value) is not None)
        ]
        for _, name_row in names.iterrows():
            candidates: list[tuple[float, pd.Series]] = []
            for _, number_row in numbers.iterrows():
                dx = float(name_row["center_x_norm"]) - float(number_row["center_x_norm"])
                dy = abs(float(name_row["center_y_norm"]) - float(number_row["center_y_norm"]))
                if 0.015 <= dx <= 0.25 and dy <= 0.022:
                    candidates.append((5 * dy + dx, number_row))
            if not candidates:
                continue
            number_row = min(candidates, key=lambda item: item[0])[1]
            label_x1 = (
                float(name_row["x1"]) / float(name_row["frame_width"])
                if has_boxes and float(name_row["frame_width"]) > 0
                else float(name_row["center_x_norm"])
            )
            observations.append(
                PairObservation(
                    int(str(number_row["text"])), str(name_row["text"]),
                    float(number_row["score"]), float(name_row["score"]), int(frame_index),
                    timestamp, float(number_row["center_x_norm"]),
                    float(name_row["center_y_norm"]), label_x1, "right",
                )
            )
    return observations


def group_pair_observations(observations: list[PairObservation]) -> list[list[PairObservation]]:
    groups: list[list[PairObservation]] = []
    for observation in observations:
        for group in groups:
            representative = group[0]
            same_player = observation.shirt_number == representative.shirt_number and similarity(
                observation.player_name, representative.player_name
            ) >= 0.84
            if same_player:
                group.append(observation)
                break
        else:
            groups.append([observation])
    return groups


def select_table_pairs(
    observations: list[PairObservation], expected_players: int,
) -> list[list[PairObservation]]:
    ranked = sorted(
        group_pair_observations(observations),
        key=lambda group: (
            len({observation.frame_index for observation in group}),
            sum(
                math.sqrt(observation.number_score * observation.name_score)
                for observation in group
            ),
        ),
        reverse=True,
    )
    selected: list[list[PairObservation]] = []
    used_numbers: set[int] = set()
    for repeated in (True, False):
        for group in ranked:
            representative = group[0]
            evidence_frames = {observation.frame_index for observation in group}
            if (repeated and len(evidence_frames) < 2) or (
                not repeated and len(evidence_frames) != 1
            ):
                continue
            confidence = np.mean(
                [math.sqrt(item.number_score * item.name_score) for item in group]
            )
            if not repeated and confidence < 0.90:
                continue
            center_y = float(np.median([item.center_y for item in group]))
            duplicate = representative.shirt_number in used_numbers or any(
                similarity(representative.player_name, chosen[0].player_name) >= 0.84
                and abs(center_y - float(np.median([item.center_y for item in chosen]))) <= 0.02
                for chosen in selected
            )
            if duplicate:
                continue
            selected.append(group)
            used_numbers.add(representative.shirt_number)
            if len(selected) == expected_players:
                return selected
    return selected


def split_table_events(
    observations: list[PairObservation], max_gap_seconds: float,
    signature_threshold: float,
) -> list[list[PairObservation]]:
    """Group repeated table rows into consecutive team presentations."""
    frames: dict[int, list[PairObservation]] = {}
    for observation in observations:
        frames.setdefault(observation.frame_index, []).append(observation)
    ordered = sorted(frames.values(), key=lambda rows: rows[0].timestamp_seconds)
    events: list[list[PairObservation]] = []
    for frame in ordered:
        if not events:
            events.append(frame.copy())
            continue
        previous_index = events[-1][-1].frame_index
        previous = [row for row in events[-1] if row.frame_index == previous_index]
        used: set[int] = set()
        matches = 0
        for row in frame:
            match = next(
                (
                    index for index, other in enumerate(previous)
                    if index not in used
                    and row.shirt_number == other.shirt_number
                    and similarity(row.player_name, other.player_name) >= 0.72
                ),
                None,
            )
            if match is not None:
                used.add(match)
                matches += 1
        similarity_score = matches / max(1, min(len(frame), len(previous)))
        gap = frame[0].timestamp_seconds - previous[0].timestamp_seconds
        if gap <= max_gap_seconds and similarity_score >= signature_threshold:
            events[-1].extend(frame)
        else:
            events.append(frame.copy())
    return events


def _table_records(
    segment: pd.DataFrame, selected: list[list[PairObservation]], lineup_index: int,
) -> list[dict[str, object]]:
    selected.sort(
        key=lambda group: float(np.median([observation.center_y for observation in group]))
    )
    records: list[dict[str, object]] = []
    for slot_index, group in enumerate(selected, start=1):
        name_observations = [
            (
                observation.player_name,
                observation.name_score,
                observation.frame_index,
            )
            for observation in group
        ]
        player_name, name_confidence, name_evidence = consensus_text(name_observations)
        number_confidence = np.mean([observation.number_score for observation in group])
        pair_confidence = np.mean(
            [math.sqrt(observation.number_score * observation.name_score) for observation in group]
        )
        records.append(
            {
                "video": str(segment["video"].iloc[0]), "segment_index": int(segment["segment_index"].iloc[0]),
                "lineup_index": lineup_index, "resolution_method": "table",
                "formation_timestamp_seconds": min(
                    observation.timestamp_seconds for observation in group
                ),
                "slot_index": slot_index, "row_index": slot_index,
                "shirt_number": group[0].shirt_number,
                "formation_label": player_name, "player_name": player_name,
                "number_confidence": round(number_confidence, 6),
                "label_confidence": round(name_confidence, 6), "name_confidence": round(name_confidence, 6),
                "pair_confidence": round(pair_confidence, 6),
                "number_evidence_frames": len({observation.frame_index for observation in group}),
                "label_evidence_frames": name_evidence, "full_name_evidence_frames": name_evidence,
                "slot_center_x_norm": round(float(np.median([item.center_x for item in group])), 6),
                "slot_center_y_norm": round(float(np.median([item.center_y for item in group])), 6),
            }
        )
    return records


def resolve_table_events(
    segment: pd.DataFrame, expected_players: int, max_gap_seconds: float,
    signature_threshold: float,
) -> tuple[list[dict[str, object]], int]:
    """Resolve every complete table/list event in a clip."""
    events = split_table_events(
        table_pair_observations(segment), max_gap_seconds, signature_threshold
    )
    records: list[dict[str, object]] = []
    best_count = 0
    for event in events:
        selected = select_table_pairs(event, expected_players)
        best_count = max(best_count, len(selected))
        if len(selected) != expected_players:
            continue
        lineup_index = len(records) // expected_players + 1
        records.extend(_table_records(segment, selected, lineup_index))
    return records, best_count


def table_rows_for_frame(
    frame: pd.DataFrame, observations: list[PairObservation], expected_players: int,
) -> list[pd.Series]:
    if not observations or not {"x1", "frame_width"}.issubset(frame.columns):
        return []
    anchor_x1 = float(np.median([observation.label_x1 for observation in observations]))
    candidates: list[pd.Series] = []
    for _, row in frame.iterrows():
        inline = parse_inline_player(row["text"])
        if row["text_type"] == "shirt_number_candidate" or (
            inline is None and not is_table_name_like(row["text"])
        ):
            continue
        frame_width = float(row["frame_width"])
        if frame_width <= 0:
            continue
        x1_normalized = float(row["x1"]) / frame_width
        if abs(x1_normalized - anchor_x1) <= 0.05:
            candidates.append(row)
    candidates.sort(key=lambda row: float(row["center_y_norm"]))
    deduplicated: list[pd.Series] = []
    for row in candidates:
        if (
            not deduplicated
            or float(row["center_y_norm"]) - float(deduplicated[-1]["center_y_norm"]) > 0.02
        ):
            deduplicated.append(row)
        elif float(row["score"]) > float(deduplicated[-1]["score"]):
            deduplicated[-1] = row
    if len(deduplicated) < expected_players:
        return []
    best_rows: list[pd.Series] = []
    best_regularity = math.inf
    for start in range(len(deduplicated) - expected_players + 1):
        rows = deduplicated[start : start + expected_players]
        y_values = [float(row["center_y_norm"]) for row in rows]
        gaps = np.diff(y_values)
        median_gap = float(np.median(gaps))
        if not 0.03 <= median_gap <= 0.085:
            continue
        regularity = float(np.mean(np.abs(gaps - median_gap)))
        if regularity < best_regularity:
            best_regularity = regularity
            best_rows = rows
    return best_rows
