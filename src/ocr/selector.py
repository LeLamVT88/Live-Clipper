"""Select complete lineup frames from sparse scout OCR."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .frames import group_records_by_segment
from .resolver.common import (
    SubstitutePanel,
    find_substitute_panel,
    is_table_name_like,
    normalize_text,
    parse_inline_player,
    shirt_number_rows,
)
from .resolver.layout import formation_anchor_pairs
from .resolver.table import table_pair_observations


SEGMENT_KEY_COLUMNS = ["video", "segment_index"]


class LineupFrameSelectionError(Exception):
    """Raised when selected lineup frames cannot be prepared."""


@dataclass(frozen=True)
class FrameCandidate:
    video: str
    segment_index: int
    frame_index: int
    timestamp_seconds: float
    layout: str
    score: float
    formation_anchor_count: int
    table_pair_count: int
    number_count: int
    name_count: int
    crop_x1_norm: float
    crop_x2_norm: float

    @property
    def crop_side(self) -> str:
        if self.crop_x1_norm > 0:
            return "right"
        if self.crop_x2_norm < 1:
            return "left"
        return "full"


@dataclass(frozen=True)
class SegmentSelection:
    video: str
    segment_index: int
    layout: str
    status: str
    score: float
    scout_frame_index: int
    scout_timestamp_seconds: float
    formation_anchor_count: int
    table_pair_count: int
    number_count: int
    name_count: int
    crop_x1_norm: float
    crop_x2_norm: float
    selected_frame_indices: tuple[int, ...]
    message: str

    @property
    def crop_side(self) -> str:
        if self.crop_x1_norm > 0:
            return "right"
        if self.crop_x2_norm < 1:
            return "left"
        return "full"


def sample_scout_frames(
    frame_records: list[dict[str, object]],
    scout_fps: float,
) -> list[dict[str, object]]:
    if scout_fps <= 0:
        raise LineupFrameSelectionError("scout_fps must be positive.")

    period_seconds = 1.0 / scout_fps
    grouped = group_records_by_segment(frame_records)

    selected: list[dict[str, object]] = []
    for records in grouped.values():
        ordered = sorted(records, key=lambda row: float(row["timestamp_seconds"]))
        next_timestamp = -float("inf")
        for record in ordered:
            timestamp = float(record["timestamp_seconds"])
            if timestamp + 1e-6 < next_timestamp:
                continue
            selected.append(record)
            next_timestamp = timestamp + period_seconds
    return selected


def crop_detections_to_formation(
    frame: pd.DataFrame,
    panel: SubstitutePanel | None,
    timestamp_seconds: float,
) -> tuple[pd.DataFrame, float, float]:
    if panel is None or timestamp_seconds < panel.first_timestamp_seconds:
        return frame, 0.0, 1.0
    if panel.side == "left":
        return (
            frame[frame["center_x_norm"] >= panel.boundary],
            panel.boundary,
            1.0,
        )
    return (
        frame[frame["center_x_norm"] <= panel.boundary],
        0.0,
        panel.boundary,
    )


def unique_table_pair_count(frame: pd.DataFrame) -> int:
    pairs = {
        (observation.shirt_number, normalize_text(observation.player_name))
        for observation in table_pair_observations(frame)
    }
    return len(pairs)


def score_frame_candidate(
    frame: pd.DataFrame,
    panel: SubstitutePanel | None,
) -> FrameCandidate | None:
    if frame.empty:
        return None

    video = str(frame["video"].iloc[0])
    segment_index = int(frame["segment_index"].iloc[0])
    frame_index = int(frame["frame_index"].iloc[0])
    timestamp = float(frame["timestamp_seconds"].iloc[0])
    formation_frame, crop_x1, crop_x2 = crop_detections_to_formation(
        frame,
        panel,
        timestamp_seconds=timestamp,
    )
    anchors = formation_anchor_pairs(formation_frame)
    number_count = len(shirt_number_rows(formation_frame))
    name_count = sum(
        1
        for text in formation_frame["text"]
        if is_table_name_like(text) and parse_inline_player(text) is None
    )
    panel_active = (
        panel is not None and timestamp >= panel.first_timestamp_seconds
    )
    table_pair_count = 0 if panel_active else unique_table_pair_count(frame)

    if panel_active and len(anchors) >= 3:
        layout = f"formation_substitutes_{panel.side}"
    elif table_pair_count >= 6 and table_pair_count >= len(anchors):
        layout = "starter_list"
    elif len(anchors) >= 3:
        layout = "formation"
    else:
        return None

    normalized_text = {
        normalize_text(text) for text in formation_frame["text"]
    }
    title_bonus = 4 if any(
        title in normalized_text
        for title in ("team lineup", "lineup", "starting xi")
    ) else 0
    panel_bonus = 8 if panel_active else 0
    score = (
        5 * len(anchors)
        + 4 * table_pair_count
        + 1.5 * min(number_count, 11)
        + 0.5 * min(name_count, 11)
        + title_bonus
        + panel_bonus
    )
    return FrameCandidate(
        video=video,
        segment_index=segment_index,
        frame_index=frame_index,
        timestamp_seconds=timestamp,
        layout=layout,
        score=round(float(score), 3),
        formation_anchor_count=len(anchors),
        table_pair_count=table_pair_count,
        number_count=number_count,
        name_count=name_count,
        crop_x1_norm=round(crop_x1, 6),
        crop_x2_norm=round(crop_x2, 6),
    )


def choose_centered_frame_indices(
    frame_records: list[dict[str, object]],
    best_frame_index: int,
    count: int,
) -> tuple[int, ...]:
    if count <= 0:
        raise LineupFrameSelectionError("selected frame count must be positive.")

    ordered = sorted(
        frame_records,
        key=lambda row: (
            float(row["timestamp_seconds"]),
            int(row["frame_index"]),
        ),
    )
    if not ordered:
        raise LineupFrameSelectionError(
            "Cannot select OCR frames from an empty segment."
        )

    best_position = next(
        (
            index
            for index, record in enumerate(ordered)
            if int(record["frame_index"]) == best_frame_index
        ),
        None,
    )
    if best_position is None:
        raise LineupFrameSelectionError(
            f"Best scout frame metadata is missing: {best_frame_index}"
        )

    radius = count // 2
    start = max(0, best_position - radius)
    end = min(len(ordered), start + count)
    start = max(0, end - count)
    return tuple(
        sorted(
            int(record["frame_index"])
            for record in ordered[start:end]
        )
    )


def fallback_selection(
    key: tuple[str, int],
    records: list[dict[str, object]],
) -> SegmentSelection:
    indices = tuple(
        sorted(int(record["frame_index"]) for record in records)
    )
    first_timestamp = min(
        float(record["timestamp_seconds"]) for record in records
    )
    return SegmentSelection(
        video=key[0],
        segment_index=key[1],
        layout="fallback_full_segment",
        status="fallback",
        score=0.0,
        scout_frame_index=0,
        scout_timestamp_seconds=first_timestamp,
        formation_anchor_count=0,
        table_pair_count=0,
        number_count=0,
        name_count=0,
        crop_x1_norm=0.0,
        crop_x2_norm=1.0,
        selected_frame_indices=indices,
        message=(
            "No complete formation/list candidate found in scout OCR; "
            "using legacy full-segment OCR."
        ),
    )


def stable_candidate_score(
    candidate: FrameCandidate,
    candidates: list[FrameCandidate],
    scout_period: float,
) -> tuple[float, float, int]:
    neighbors = sum(
        1
        for other in candidates
        if (
            other.layout == candidate.layout
            and abs(
                other.timestamp_seconds - candidate.timestamp_seconds
            )
            <= 2.1 * scout_period
        )
    )
    return (
        candidate.score + 3 * min(neighbors, 4),
        candidate.timestamp_seconds,
        candidate.frame_index,
    )


def selected_candidate(
    best: FrameCandidate,
    selected_indices: tuple[int, ...],
    stable_score: float,
) -> SegmentSelection:
    return SegmentSelection(
        video=best.video,
        segment_index=best.segment_index,
        layout=best.layout,
        status="selected",
        score=stable_score,
        scout_frame_index=best.frame_index,
        scout_timestamp_seconds=best.timestamp_seconds,
        formation_anchor_count=best.formation_anchor_count,
        table_pair_count=best.table_pair_count,
        number_count=best.number_count,
        name_count=best.name_count,
        crop_x1_norm=best.crop_x1_norm,
        crop_x2_norm=best.crop_x2_norm,
        selected_frame_indices=selected_indices,
        message=(
            f"Selected {len(selected_indices)} detailed OCR frame(s) "
            f"in a centered window around scout frame {best.frame_index}."
        ),
    )


def select_segment_frames(
    frame_records: list[dict[str, object]],
    scout_detections: pd.DataFrame,
    selected_frame_count: int,
    scout_fps: float,
) -> list[SegmentSelection]:
    if selected_frame_count <= 0:
        raise LineupFrameSelectionError(
            "selected_frame_count must be positive."
        )
    if scout_fps <= 0:
        raise LineupFrameSelectionError("scout_fps must be positive.")

    records_by_segment = group_records_by_segment(frame_records)

    detections_by_segment = (
        {
            (str(video), int(segment_index)): group
            for (video, segment_index), group in scout_detections.groupby(
                SEGMENT_KEY_COLUMNS,
                sort=False,
            )
        }
        if set(SEGMENT_KEY_COLUMNS).issubset(scout_detections.columns)
        else {}
    )
    selections: list[SegmentSelection] = []
    scout_period = 1.0 / scout_fps

    for key, segment_records in records_by_segment.items():
        segment_detections = detections_by_segment.get(key, pd.DataFrame())
        panel = find_substitute_panel(segment_detections)
        candidates: list[FrameCandidate] = []
        if not segment_detections.empty:
            for _, frame in segment_detections.groupby(
                "frame_index",
                sort=True,
            ):
                candidate = score_frame_candidate(frame, panel)
                if candidate is not None:
                    candidates.append(candidate)

        if not candidates:
            selections.append(
                fallback_selection(key, segment_records)
            )
            continue

        score = lambda candidate: stable_candidate_score(
            candidate,
            candidates,
            scout_period,
        )
        best = max(candidates, key=score)
        selected_indices = choose_centered_frame_indices(
            segment_records,
            best_frame_index=best.frame_index,
            count=selected_frame_count,
        )
        selections.append(
            selected_candidate(
                best,
                selected_indices,
                stable_score=score(best)[0],
            )
        )

    return selections
