from __future__ import annotations

from dataclasses import asdict, dataclass

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


class LineupFrameSelectionError(Exception): pass


@dataclass(frozen=True, slots=True)
class CandidateStats:
    video: str
    segment_index: int
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


@dataclass(frozen=True, slots=True)
class FrameCandidate(CandidateStats):
    frame_index: int
    timestamp_seconds: float


@dataclass(frozen=True, slots=True)
class SegmentSelection(CandidateStats):
    status: str
    scout_frame_index: int
    scout_timestamp_seconds: float
    selected_frame_indices: tuple[int, ...]
    message: str


def sample_scout_frames(frame_records: list[dict[str, object]], scout_fps: float) -> list[dict[str, object]]:
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


def score_frame_candidate(frame: pd.DataFrame, panel: SubstitutePanel | None) -> FrameCandidate | None:
    if frame.empty:
        return None
    video = str(frame["video"].iloc[0])
    segment_index = int(frame["segment_index"].iloc[0])
    frame_index = int(frame["frame_index"].iloc[0])
    timestamp = float(frame["timestamp_seconds"].iloc[0])
    panel_active = panel is not None and timestamp >= panel.first_timestamp_seconds
    if not panel_active:
        formation_frame, crop_x1, crop_x2 = frame, 0.0, 1.0
    elif panel.side == "left":
        formation_frame, crop_x1, crop_x2 = (
            frame[frame["center_x_norm"] >= panel.boundary], panel.boundary, 1.0
        )
    else:
        formation_frame, crop_x1, crop_x2 = (
            frame[frame["center_x_norm"] <= panel.boundary], 0.0, panel.boundary
        )
    anchors = formation_anchor_pairs(formation_frame)
    number_count = len(shirt_number_rows(formation_frame))
    name_count = sum(
        1
        for text in formation_frame["text"]
        if is_table_name_like(text) and parse_inline_player(text) is None
    )
    table_pair_count = 0 if panel_active else len({
        (item.shirt_number, normalize_text(item.player_name))
        for item in table_pair_observations(frame)
    })
    if panel_active and len(anchors) >= 3:
        layout = f"formation_substitutes_{panel.side}"
    elif table_pair_count >= 6 and table_pair_count >= len(anchors):
        layout = "starter_list"
    elif len(anchors) >= 3:
        layout = "formation"
    else:
        return None
    normalized_text = {normalize_text(text) for text in formation_frame["text"]}
    titles = "team lineup", "lineup", "starting xi"
    title_bonus = 4 if any(title in normalized_text for title in titles) else 0
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
        video=video, segment_index=segment_index, frame_index=frame_index,
        timestamp_seconds=timestamp, layout=layout, score=round(float(score), 3),
        formation_anchor_count=len(anchors), table_pair_count=table_pair_count,
        number_count=number_count, name_count=name_count,
        crop_x1_norm=round(crop_x1, 6), crop_x2_norm=round(crop_x2, 6),
    )


def choose_spread_frame_indices(
    frame_records: list[dict[str, object]], candidates: list[FrameCandidate],
    best: FrameCandidate, count: int, scout_period_seconds: float,
) -> tuple[int, ...]:
    if count <= 0:
        raise LineupFrameSelectionError("selected frame count must be positive.")
    same_layout = sorted(
        (candidate for candidate in candidates if candidate.layout == best.layout),
        key=lambda candidate: candidate.timestamp_seconds,
    )
    best_position = next(
        i for i, candidate in enumerate(same_layout) if candidate.frame_index == best.frame_index
    )
    # Sparse scout OCR can miss one otherwise stable lineup frame.  Allow a
    # little over two scout periods so the detail frames still span the same
    # graphic scene instead of clustering around the best frame.
    maximum_gap = 2.1 * scout_period_seconds
    start = best_position
    while (
        start > 0
        and same_layout[start].timestamp_seconds - same_layout[start - 1].timestamp_seconds
        <= maximum_gap
    ):
        start -= 1
    end = best_position
    while (
        end + 1 < len(same_layout)
        and same_layout[end + 1].timestamp_seconds - same_layout[end].timestamp_seconds
        <= maximum_gap
    ):
        end += 1
    scene = same_layout[start : end + 1]
    if len(scene) <= count:
        selected_indices = {candidate.frame_index for candidate in scene}
    elif count == 1:
        selected_indices = {best.frame_index}
    else:
        positions = [round(position * (len(scene) - 1) / (count - 1)) for position in range(count)]
        selected_indices = {scene[position].frame_index for position in positions}
    if len(selected_indices) >= count:
        return tuple(sorted(selected_indices))
    nearest = sorted(
        frame_records,
        key=lambda row: (
            abs(float(row["timestamp_seconds"]) - best.timestamp_seconds),
            int(row["frame_index"]),
        ),
    )
    for record in nearest:
        selected_indices.add(int(record["frame_index"]))
        if len(selected_indices) == count:
            break
    return tuple(sorted(selected_indices))


def fallback_selection(
    key: tuple[str, int], records: list[dict[str, object]],
    message: str = (
        "No complete formation/list candidate found in scout OCR; "
        "using legacy full-segment OCR."
    ),
) -> SegmentSelection:
    indices = tuple(sorted(int(record["frame_index"]) for record in records))
    first_timestamp = min(float(record["timestamp_seconds"]) for record in records)
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
        message=message,
    )


def select_segment_frames(
    frame_records: list[dict[str, object]], scout_detections: pd.DataFrame,
    selected_frame_count: int, scout_fps: float,
) -> list[SegmentSelection]:
    if selected_frame_count <= 0:
        raise LineupFrameSelectionError("selected_frame_count must be positive.")
    if scout_fps <= 0:
        raise LineupFrameSelectionError("scout_fps must be positive.")
    records_by_segment = group_records_by_segment(frame_records)
    detections_by_segment = (
        {
            (str(video), int(segment_index)): group
            for (video, segment_index), group in scout_detections.groupby(SEGMENT_KEY_COLUMNS, sort=False)
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
            for _, frame in segment_detections.groupby("frame_index", sort=True):
                candidate = score_frame_candidate(frame, panel)
                if candidate is not None:
                    candidates.append(candidate)
        if not candidates:
            selections.append(fallback_selection(key, segment_records))
            continue
        def score(candidate: FrameCandidate) -> tuple[float, float, int]:
            neighbors = sum(
                other.layout == candidate.layout
                and abs(other.timestamp_seconds - candidate.timestamp_seconds) <= 2.1 * scout_period
                for other in candidates
            )
            return (
                candidate.score + 3 * min(neighbors, 4),
                candidate.timestamp_seconds, candidate.frame_index,
            )
        best = max(candidates, key=score)
        selected_indices = choose_spread_frame_indices(
            segment_records,
            candidates=candidates,
            best=best,
            count=selected_frame_count,
            scout_period_seconds=scout_period,
        )
        data = asdict(best)
        frame_index, timestamp = data.pop("frame_index"), data.pop("timestamp_seconds")
        data["score"] = score(best)[0]
        selections.append(SegmentSelection(
            **data, status="selected", scout_frame_index=frame_index,
            scout_timestamp_seconds=timestamp, selected_frame_indices=selected_indices,
            message=(f"Selected {len(selected_indices)} detailed OCR frame(s) across the stable "
                     f"scene containing scout frame {best.frame_index}."),
        ))
    return selections
