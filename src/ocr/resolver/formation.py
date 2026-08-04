"""Resolve number-name slots from formation graphics."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from .common import (
    FormationEvent,
    LineupResolutionError,
    NumberCluster,
    consensus_text,
    detections_without_substitute_panel,
    is_name_like,
    normalize_text,
    shirt_number_rows,
    spatial_distance,
)
from .player_names import best_full_name


def signature_similarity(
    left: list[pd.Series],
    right: list[pd.Series],
    x_tolerance: float = 0.045,
    y_tolerance: float = 0.065,
) -> float:
    if not left or not right:
        return 0.0

    matched_right: set[int] = set()
    matches = 0
    for left_row in left:
        best_index: int | None = None
        best_distance = math.inf
        for index, right_row in enumerate(right):
            if (
                index in matched_right
                or str(left_row["text"])
                != str(right_row["text"])
            ):
                continue
            dx, dy = spatial_distance(left_row, right_row)
            if dx <= x_tolerance and dy <= y_tolerance:
                distance = dx + dy
                if distance < best_distance:
                    best_distance = distance
                    best_index = index
        if best_index is not None:
            matched_right.add(best_index)
            matches += 1

    return matches / max(1, min(len(left), len(right)))


def detect_formation_events(
    segment: pd.DataFrame,
    min_number_count: int,
    max_gap_seconds: float,
    signature_threshold: float,
) -> list[FormationEvent]:
    snapshot_candidates: list[
        tuple[int, float, list[pd.Series]]
    ] = []
    numbers = shirt_number_rows(segment)

    for frame_index, frame in numbers.groupby(
        "frame_index",
        sort=True,
    ):
        if len(frame) < min_number_count:
            continue
        timestamp = float(frame["timestamp_seconds"].iloc[0])
        signature = [row for _, row in frame.iterrows()]
        snapshot_candidates.append(
            (int(frame_index), timestamp, signature)
        )

    events: list[FormationEvent] = []
    for frame_index, timestamp, signature in snapshot_candidates:
        if not events:
            events.append(
                FormationEvent(
                    [frame_index],
                    [timestamp],
                    [signature],
                )
            )
            continue

        current = events[-1]
        time_gap = timestamp - current.end_seconds
        best_similarity = max(
            signature_similarity(signature, existing)
            for existing in current.signatures
        )
        if (
            time_gap <= max_gap_seconds
            and best_similarity >= signature_threshold
        ):
            current.snapshot_frames.append(frame_index)
            current.snapshot_timestamps.append(timestamp)
            current.signatures.append(signature)
        else:
            events.append(
                FormationEvent(
                    [frame_index],
                    [timestamp],
                    [signature],
                )
            )

    return events


def cluster_number_positions(
    event: FormationEvent,
    segment: pd.DataFrame,
    x_tolerance: float = 0.045,
    y_tolerance: float = 0.065,
) -> list[NumberCluster]:
    snapshot_numbers = shirt_number_rows(
        segment[
            segment["frame_index"].isin(event.snapshot_frames)
        ]
    ).sort_values(
        ["frame_index", "center_y_norm", "center_x_norm"]
    )
    clusters: list[NumberCluster] = []

    for _, observation in snapshot_numbers.iterrows():
        candidates: list[tuple[float, NumberCluster]] = []
        for cluster in clusters:
            dx = abs(
                float(observation["center_x_norm"])
                - cluster.center_x
            )
            dy = abs(
                float(observation["center_y_norm"])
                - cluster.center_y
            )
            if dx <= x_tolerance and dy <= y_tolerance:
                candidates.append((dx + dy, cluster))
        if candidates:
            min(
                candidates,
                key=lambda item: item[0],
            )[1].observations.append(observation)
        else:
            clusters.append(
                NumberCluster(observations=[observation])
            )

    return clusters


def nearby_formation_labels(
    cluster: NumberCluster,
    event: FormationEvent,
    segment: pd.DataFrame,
) -> list[tuple[str, float, int]]:
    text_rows = segment[
        segment["frame_index"].isin(event.snapshot_frames)
        & (segment["text_type"] != "shirt_number_candidate")
    ]
    labels: list[tuple[str, float, int]] = []

    for frame_index, frame in text_rows.groupby(
        "frame_index",
        sort=False,
    ):
        candidates: list[tuple[float, pd.Series]] = []
        for _, row in frame.iterrows():
            if not is_name_like(row["text"]):
                continue
            dx = abs(
                float(row["center_x_norm"]) - cluster.center_x
            )
            dy = (
                float(row["center_y_norm"]) - cluster.center_y
            )
            # Compact player cards (for example the Serie A graphic) place
            # the label only ~0.027 image-heights below the shirt number.
            # Keep this aligned with formation_anchor_pairs(), which accepts
            # the same compact geometry from 0.025.
            if dx <= 0.07 and 0.025 <= dy <= 0.13:
                candidates.append(
                    (dx + abs(dy - 0.078), row)
                )
        if candidates:
            row = min(
                candidates,
                key=lambda item: item[0],
            )[1]
            labels.append(
                (
                    str(row["text"]),
                    float(row["score"]),
                    int(frame_index),
                )
            )

    return labels


def cluster_number_consensus(
    cluster: NumberCluster,
) -> tuple[int, float, int]:
    by_number: defaultdict[
        int,
        list[pd.Series],
    ] = defaultdict(list)
    for row in cluster.observations:
        by_number[int(str(row["text"]))].append(row)

    best_number = max(
        by_number,
        key=lambda number: (
            len(
                {
                    int(row["frame_index"])
                    for row in by_number[number]
                }
            ),
            sum(
                float(row["score"])
                for row in by_number[number]
            ),
        ),
    )
    rows = by_number[best_number]
    return (
        best_number,
        float(
            np.mean(
                [float(row["score"]) for row in rows]
            )
        ),
        len({int(row["frame_index"]) for row in rows}),
    )


def select_player_clusters(
    clusters: list[NumberCluster],
    event: FormationEvent,
    segment: pd.DataFrame,
    expected_players: int,
) -> list[NumberCluster]:
    for cluster in clusters:
        cluster.label_observations = nearby_formation_labels(
            cluster,
            event,
            segment,
        )

    ranked = sorted(
        clusters,
        key=lambda cluster: (
            bool(cluster.label_observations),
            len(
                {
                    frame
                    for _, _, frame
                    in cluster.label_observations
                }
            ),
            cluster.frame_count,
            sum(
                float(row["score"])
                for row in cluster.observations
            ),
        ),
        reverse=True,
    )
    selected = ranked[:expected_players]
    return sorted(
        selected,
        key=lambda cluster: (
            cluster.center_y,
            cluster.center_x,
        ),
    )


def formation_rows(
    clusters: list[NumberCluster],
) -> dict[int, int]:
    row_by_cluster: dict[int, int] = {}
    current_row = 0
    previous_y: float | None = None
    for cluster in clusters:
        if (
            previous_y is None
            or cluster.center_y - previous_y > 0.09
        ):
            current_row += 1
        row_by_cluster[id(cluster)] = current_row
        previous_y = cluster.center_y
    return row_by_cluster


def validate_unique_shirt_numbers(
    records: list[dict[str, object]],
    lineup_index: int,
) -> None:
    counts: defaultdict[int, int] = defaultdict(int)
    for record in records:
        counts[int(record["shirt_number"])] += 1
    duplicates = sorted(
        number
        for number, count in counts.items()
        if count > 1
    )
    if duplicates:
        duplicate_text = ", ".join(
            str(number) for number in duplicates
        )
        raise LineupResolutionError(
            f"Duplicate shirt number(s) {duplicate_text} "
            f"found for lineup {lineup_index}."
        )


def resolve_event(
    event: FormationEvent,
    segment: pd.DataFrame,
    event_end_seconds: float,
    lineup_index: int,
    expected_players: int,
) -> list[dict[str, object]]:
    clusters = cluster_number_positions(event, segment)
    selected = select_player_clusters(
        clusters,
        event,
        segment,
        expected_players=expected_players,
    )
    if len(selected) < expected_players:
        raise LineupResolutionError(
            f"Only {len(selected)} player slots found for "
            f"lineup {lineup_index}; expected {expected_players}."
        )

    row_indices = formation_rows(selected)
    selected = sorted(
        selected,
        key=lambda cluster: (
            row_indices[id(cluster)],
            cluster.center_x,
        ),
    )
    label_results = {
        id(cluster): consensus_text(
            cluster.label_observations
        )
        for cluster in selected
    }
    missing_labels = [
        cluster
        for cluster in selected
        if not label_results[id(cluster)][0]
    ]
    if missing_labels:
        raise LineupResolutionError(
            f"{len(missing_labels)} player slot(s) have no "
            f"nearby name label for lineup {lineup_index}."
        )

    event_detections = segment[
        (
            segment["timestamp_seconds"]
            >= event.start_seconds
        )
        & (
            segment["timestamp_seconds"]
            < event_end_seconds
        )
    ]
    normalized_labels = {
        normalize_text(label_results[id(cluster)][0])
        for cluster in selected
    }
    records: list[dict[str, object]] = []

    for slot_index, cluster in enumerate(
        selected,
        start=1,
    ):
        (
            shirt_number,
            number_confidence,
            number_evidence,
        ) = cluster_number_consensus(cluster)
        (
            formation_label,
            label_confidence,
            label_evidence,
        ) = label_results[id(cluster)]
        other_labels = normalized_labels - {
            normalize_text(formation_label)
        }
        (
            player_name,
            name_confidence,
            name_evidence,
        ) = best_full_name(
            formation_label,
            event_detections,
            other_formation_labels=other_labels,
        )
        pair_confidence = (
            number_confidence
            * label_confidence
            * max(name_confidence, 0.01)
        ) ** (1 / 3)
        records.append(
            {
                "video": str(segment["video"].iloc[0]),
                "segment_index": int(
                    segment["segment_index"].iloc[0]
                ),
                "lineup_index": lineup_index,
                "resolution_method": "formation",
                "formation_timestamp_seconds": (
                    event.start_seconds
                ),
                "slot_index": slot_index,
                "row_index": row_indices[id(cluster)],
                "shirt_number": shirt_number,
                "formation_label": formation_label,
                "player_name": player_name,
                "number_confidence": round(
                    number_confidence,
                    6,
                ),
                "label_confidence": round(
                    label_confidence,
                    6,
                ),
                "name_confidence": round(
                    name_confidence,
                    6,
                ),
                "pair_confidence": round(
                    pair_confidence,
                    6,
                ),
                "number_evidence_frames": number_evidence,
                "label_evidence_frames": label_evidence,
                "full_name_evidence_frames": name_evidence,
                "slot_center_x_norm": round(
                    cluster.center_x,
                    6,
                ),
                "slot_center_y_norm": round(
                    cluster.center_y,
                    6,
                ),
            }
        )

    validate_unique_shirt_numbers(
        records,
        lineup_index=lineup_index,
    )
    return records


def attempt_formation_resolution(
    segment: pd.DataFrame,
    expected_players: int,
    min_number_count: int,
    max_gap_seconds: float,
    signature_threshold: float,
) -> tuple[
    pd.DataFrame,
    list[FormationEvent],
    list[dict[str, object]],
    list[str],
]:
    formation_segment = detections_without_substitute_panel(
        segment
    )
    events = detect_formation_events(
        formation_segment,
        min_number_count=min_number_count,
        max_gap_seconds=max_gap_seconds,
        signature_threshold=signature_threshold,
    )
    segment_records: list[dict[str, object]] = []
    event_errors: list[str] = []
    if not events:
        return (
            formation_segment,
            events,
            segment_records,
            event_errors,
        )

    segment_end = float(
        formation_segment["segment_end_seconds"].max()
    )
    for index, event in enumerate(events):
        event_end = (
            events[index + 1].start_seconds
            if index + 1 < len(events)
            else segment_end + 1e-6
        )
        resolved_lineup_index = (
            len(segment_records) // expected_players + 1
        )
        try:
            segment_records.extend(
                resolve_event(
                    event,
                    formation_segment,
                    event_end_seconds=event_end,
                    lineup_index=resolved_lineup_index,
                    expected_players=expected_players,
                )
            )
        except LineupResolutionError as exc:
            event_errors.append(str(exc))
    return (
        formation_segment,
        events,
        segment_records,
        event_errors,
    )
