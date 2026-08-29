"""PySceneDetect boundary snapping without a visual language model."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Literal

from .schema import LineupDetectionError, LineupDetectionResult, LineupSegment


DEFAULT_SCENE_THRESHOLD = 27.0
DEFAULT_SCENE_MIN_LENGTH_FRAMES = 15
DEFAULT_SCENE_SNAP_RADIUS_SECONDS = 4.0
DEFAULT_SOFT_TRANSITION_SCORE = 5.0
DEFAULT_MIN_GRAPHIC_DURATION_SECONDS = 18.0
DEFAULT_MAX_GRAPHIC_DURATION_SECONDS = 45.0
DEFAULT_MAX_AUDIO_END_LEAD_SECONDS = 18.0
DEFAULT_MAX_AUDIO_END_TRAIL_SECONDS = 25.0
DEFAULT_MAX_AUDIO_START_OFFSET_SECONDS = 20.0
DEFAULT_SLIDESHOW_STABLE_SCENE_SECONDS = 4.0
DEFAULT_SLIDESHOW_DURATION_TOLERANCE_SECONDS = 1.0


@dataclass(frozen=True)
class SceneSnap:
    original_seconds: float
    snapped_seconds: float
    distance_seconds: float
    applied: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "original_seconds": round(self.original_seconds, 3),
            "snapped_seconds": round(self.snapped_seconds, 3),
            "distance_seconds": round(self.distance_seconds, 3),
            "applied": self.applied,
        }


def detect_scene_cuts(
    video_path: Path,
    *,
    start_seconds: float,
    end_seconds: float,
    threshold: float = DEFAULT_SCENE_THRESHOLD,
    min_scene_len_frames: int = DEFAULT_SCENE_MIN_LENGTH_FRAMES,
) -> tuple[float, ...]:
    """Run ContentDetector only over the time range relevant to lineups."""
    source = video_path.expanduser().resolve()
    if not source.is_file():
        raise LineupDetectionError(f"Source video does not exist: {source}")
    if not all(math.isfinite(value) for value in (start_seconds, end_seconds)):
        raise LineupDetectionError("Scene detection bounds must be finite.")
    if start_seconds < 0 or end_seconds <= start_seconds:
        raise LineupDetectionError("Invalid scene detection range.")
    if threshold <= 0 or min_scene_len_frames <= 0:
        raise LineupDetectionError(
            "Scene threshold and minimum scene length must be positive."
        )
    try:
        from scenedetect import SceneManager, StatsManager, open_video
        from scenedetect.detectors import ContentDetector
    except ModuleNotFoundError as exc:
        raise LineupDetectionError(
            "scenedetect is not installed. Install requirements.txt first."
        ) from exc

    try:
        video = open_video(str(source))
        # PySceneDetect interprets ``int`` as a frame number and ``float`` as
        # seconds.  Cast explicitly so a caller passing 535 never seeks frame
        # 535 (~17.8s at 30 fps).
        video.seek(float(start_seconds))
        stats = StatsManager()
        manager = SceneManager(stats_manager=stats)
        manager.add_detector(
            ContentDetector(
                threshold=threshold,
                min_scene_len=min_scene_len_frames,
            )
        )
        manager.detect_scenes(
            video=video,
            end_time=float(end_seconds),
            show_progress=False,
        )
        scenes = manager.get_scene_list(start_in_scene=True)
    except Exception as exc:
        raise LineupDetectionError(f"PySceneDetect failed: {exc}") from exc
    hard_cuts = tuple(
        scene_start.get_seconds()
        for scene_start, _ in scenes[1:]
        if start_seconds <= scene_start.get_seconds() <= end_seconds
    )
    soft_cuts = _soft_transition_cuts(
        stats,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        frame_rate=float(video.frame_rate),
    )
    merged: list[float] = []
    for cut in sorted((*hard_cuts, *soft_cuts)):
        if merged and cut - merged[-1] < 0.15:
            continue
        merged.append(cut)
    return tuple(merged)


def _soft_transition_cuts(
    stats_manager: object,
    *,
    start_seconds: float,
    end_seconds: float,
    frame_rate: float,
    score_threshold: float = DEFAULT_SOFT_TRANSITION_SCORE,
) -> tuple[float, ...]:
    """Find the onset of a sustained dissolve using ContentDetector metrics.

    ContentDetector reports the end of a dissolve as a hard cut in some sports
    broadcasts.  Its frame metrics still expose the earlier fade onset (for
    ACLE_01 this is around 568.83s), which is the useful graphic-out boundary.
    """
    if frame_rate <= 0:
        return ()
    first_frame = max(0, int(math.floor(start_seconds * frame_rate)))
    last_frame = int(math.ceil(end_seconds * frame_rate))
    scores: dict[int, float] = {}
    for frame_number in range(first_frame, last_frame + 1):
        try:
            values = stats_manager.get_metrics(  # type: ignore[attr-defined]
                frame_number, ["content_val"]
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if values and values[0] is not None:
            scores[frame_number] = float(values[0])

    cuts: list[float] = []
    cooldown_frames = max(1, round(frame_rate * 0.5))
    previous_cut_frame = -cooldown_frames
    for frame_number in range(first_frame + 6, last_frame - 4):
        score = scores.get(frame_number)
        if score is None or score < score_threshold:
            continue
        previous = [
            scores.get(index, 0.0)
            for index in range(frame_number - 6, frame_number)
        ]
        following = [
            scores.get(index, 0.0)
            for index in range(frame_number, frame_number + 5)
        ]
        stable_before = sum(previous) / len(previous) < score_threshold * 0.5
        sustained_after = sum(value >= score_threshold for value in following) >= 3
        isolated_transition_spike = score >= max(10.0, score_threshold * 2.0)
        # Some broadcasts cut from a moving player shot to a static wide shot
        # used as the canvas for an animated lineup graphic.  Such a cut can be
        # below the normal hard-cut threshold, but the content score collapses
        # immediately afterwards (ACLE_07: 17.8 followed by ~1-2).
        previous_mean = sum(previous) / len(previous)
        following_mean = sum(following) / len(following)
        settles_into_static_shot = (
            score >= max(12.0, score_threshold * 2.4)
            and previous_mean >= score_threshold
            and following_mean <= score * 0.35
        )
        if not (
            (stable_before and (sustained_after or isolated_transition_spike))
            or settles_into_static_shot
        ):
            continue
        if frame_number - previous_cut_frame < cooldown_frames:
            continue
        cuts.append(frame_number / frame_rate)
        previous_cut_frame = frame_number

    # A lineup card is often composited gradually over an otherwise static
    # pitch shot.  The per-frame score can remain well below ContentDetector's
    # hard-cut threshold throughout that animation (ACLE_08 peaks around 6),
    # so look for a sustained low-amplitude rise after a genuinely quiet shot.
    quiet_frames = max(6, round(frame_rate * 0.6))
    transition_frames = max(8, round(frame_rate * 0.8))
    quiet_mean_max = score_threshold * 0.12
    transition_mean_min = score_threshold * 0.24
    transition_entry_min = score_threshold * 0.10
    transition_active_min = score_threshold * 0.20
    transition_peak_min = score_threshold * 0.60
    active_frame_minimum = max(3, math.ceil(transition_frames / 3))
    gentle_cuts: list[float] = []
    gentle_cooldown_frames = max(1, round(frame_rate))
    previous_gentle_frame = -gentle_cooldown_frames
    for frame_number in range(
        first_frame + quiet_frames,
        last_frame - transition_frames,
    ):
        score = scores.get(frame_number, 0.0)
        if score < transition_entry_min:
            continue
        quiet_window = [
            scores.get(index, 0.0)
            for index in range(frame_number - quiet_frames, frame_number)
        ]
        transition_window = [
            scores.get(index, 0.0)
            for index in range(
                frame_number,
                frame_number + transition_frames,
            )
        ]
        if sum(quiet_window) / len(quiet_window) > quiet_mean_max:
            continue
        if (
            sum(transition_window) / len(transition_window)
            < transition_mean_min
        ):
            continue
        if max(transition_window) < transition_peak_min:
            continue
        if (
            sum(
                value >= transition_active_min
                for value in transition_window
            )
            < active_frame_minimum
        ):
            continue
        if frame_number - previous_gentle_frame < gentle_cooldown_frames:
            continue
        gentle_cuts.append(frame_number / frame_rate)
        previous_gentle_frame = frame_number

    merged: list[float] = []
    for cut in sorted((*cuts, *gentle_cuts)):
        if merged and cut - merged[-1] < 0.5:
            continue
        merged.append(cut)
    return tuple(merged)


def snap_timestamp(
    timestamp: float,
    scene_cuts: Iterable[float],
    *,
    radius_seconds: float = DEFAULT_SCENE_SNAP_RADIUS_SECONDS,
    preference: Literal["nearest", "forward", "backward"] = "nearest",
) -> SceneSnap:
    """Use the nearest cut only when it is inside a deliberately narrow radius."""
    if not math.isfinite(timestamp) or timestamp < 0:
        raise LineupDetectionError("Timestamp to snap must be finite and non-negative.")
    if not math.isfinite(radius_seconds) or radius_seconds < 0:
        raise LineupDetectionError("Scene snap radius must be finite and non-negative.")
    if preference not in {"nearest", "forward", "backward"}:
        raise LineupDetectionError(f"Invalid scene snap preference: {preference}")
    valid_cuts = tuple(
        float(cut)
        for cut in scene_cuts
        if math.isfinite(float(cut)) and float(cut) >= 0
    )
    if not valid_cuts:
        return SceneSnap(timestamp, timestamp, 0.0, False)
    closest_overall = min(valid_cuts, key=lambda cut: (abs(cut - timestamp), cut))
    nearby = tuple(
        cut for cut in valid_cuts if abs(cut - timestamp) <= radius_seconds
    )
    if not nearby:
        return SceneSnap(
            timestamp,
            timestamp,
            abs(closest_overall - timestamp),
            False,
        )
    directional = nearby
    if preference == "forward":
        directional = tuple(cut for cut in nearby if cut >= timestamp)
    elif preference == "backward":
        directional = tuple(cut for cut in nearby if cut <= timestamp)
    pool = directional or nearby
    nearest = min(pool, key=lambda cut: (abs(cut - timestamp), cut))
    distance = abs(nearest - timestamp)
    return SceneSnap(timestamp, nearest, distance, True)


def _stable_graphic_scene(
    segment: LineupSegment,
    cuts: tuple[float, ...],
    *,
    min_graphic_duration_seconds: float,
    max_graphic_duration_seconds: float,
    max_audio_start_offset_seconds: float,
    max_audio_end_lead_seconds: float,
    max_audio_end_trail_seconds: float,
) -> tuple[float, float] | None:
    """Choose one long scene whose boundaries bracket the spoken lineup."""
    candidates: list[tuple[float, float, float]] = []
    for start, end in zip(cuts, cuts[1:]):
        duration = end - start
        if not min_graphic_duration_seconds <= duration <= max_graphic_duration_seconds:
            continue
        if abs(start - segment.start_seconds) > max_audio_start_offset_seconds:
            continue
        end_offset = end - segment.end_seconds
        max_end_offset = (
            max_audio_end_lead_seconds
            if end_offset >= 0
            else max_audio_end_trail_seconds
        )
        if abs(end_offset) > max_end_offset:
            continue
        overlap = min(end, segment.end_seconds) - max(
            start, segment.start_seconds
        )
        if overlap <= 0:
            continue
        boundary_error = abs(start - segment.start_seconds) + abs(
            end - segment.end_seconds
        )
        candidates.append((boundary_error, start, end))
    if not candidates:
        return None
    _, start, end = min(candidates, key=lambda item: (item[0], item[1]))
    return start, end


def _looks_like_consecutive_slideshow(
    result: LineupDetectionResult,
    cuts: tuple[float, ...],
    stable_scenes: tuple[tuple[float, float] | None, ...],
    *,
    max_graphic_duration_seconds: float,
) -> bool:
    if len(result.segments) < 2 or any(scene is not None for scene in stable_scenes):
        return False
    start = result.segments[0].start_seconds - max_graphic_duration_seconds
    end = result.segments[-1].start_seconds + max_graphic_duration_seconds
    relevant_cuts = tuple(cut for cut in cuts if start <= cut <= end)
    return len(relevant_cuts) >= 8


def _snap_slideshow_lineup_result(
    result: LineupDetectionResult,
    cuts: tuple[float, ...],
    *,
    radius_seconds: float,
    min_graphic_duration_seconds: float,
    max_graphic_duration_seconds: float,
    max_audio_end_lead_seconds: float,
) -> LineupDetectionResult:
    """Resolve consecutive multi-slide graphics around two roster narrations."""
    ordered_cuts = tuple(sorted(cuts))
    if len(result.segments) == 2 and ordered_cuts:
        first, second = result.segments
        first_end = snap_timestamp(
            first.end_seconds,
            ordered_cuts,
            radius_seconds=radius_seconds,
            preference="nearest",
        )
        second_start = snap_timestamp(
            second.start_seconds,
            ordered_cuts,
            radius_seconds=radius_seconds,
            preference="nearest",
        )
        first_start_candidates = tuple(
            cut
            for cut in ordered_cuts
            if first_end.snapped_seconds - max_graphic_duration_seconds
            <= cut
            <= min(
                first.start_seconds,
                first_end.snapped_seconds - min_graphic_duration_seconds,
            )
        )
        second_end_candidates = tuple(
            cut
            for cut in ordered_cuts
            if second_start.snapped_seconds + min_graphic_duration_seconds
            <= cut
            <= second_start.snapped_seconds
            + max_graphic_duration_seconds
            + DEFAULT_SLIDESHOW_DURATION_TOLERANCE_SECONDS
        )
        if (
            first_end.applied
            and second_start.applied
            and first_start_candidates
            and second_end_candidates
        ):
            first_start_seconds = min(first_start_candidates)
            second_end_seconds = max(second_end_candidates)
            ranges = (
                (first_start_seconds, first_end.snapped_seconds),
                (second_start.snapped_seconds, second_end_seconds),
            )
            refined = tuple(
                replace(
                    segment,
                    start_seconds=start,
                    end_seconds=end,
                    reason=(
                        segment.reason
                        + " PySceneDetect selected the complete multi-slide "
                        "graphic block."
                    ),
                )
                for segment, (start, end) in zip(result.segments, ranges)
            )
            diagnostics = [
                {
                    "segment_id": first.segment_id,
                    "start": SceneSnap(
                        first.start_seconds,
                        first_start_seconds,
                        abs(first_start_seconds - first.start_seconds),
                        True,
                    ).to_dict(),
                    "end": first_end.to_dict(),
                    "end_strategy": "before_next_slideshow_graphic",
                },
                {
                    "segment_id": second.segment_id,
                    "start": second_start.to_dict(),
                    "end": SceneSnap(
                        second.end_seconds,
                        second_end_seconds,
                        abs(second_end_seconds - second.end_seconds),
                        True,
                    ).to_dict(),
                    "end_strategy": "complete_slideshow_sequence",
                },
            ]
            raw_response = dict(result.raw_response)
            raw_response["scene_refinement"] = {
                "method": "pyscenedetect_content_detector",
                "mode": "consecutive_slideshow_graphics",
                "radius_seconds": radius_seconds,
                "min_graphic_duration_seconds": min_graphic_duration_seconds,
                "max_graphic_duration_seconds": max_graphic_duration_seconds,
                "max_audio_end_lead_seconds": max_audio_end_lead_seconds,
                "scene_cuts_seconds": [
                    round(cut, 3) for cut in ordered_cuts
                ],
                "segments": diagnostics,
            }
            snapped = replace(
                result,
                segments=refined,
                raw_response=raw_response,
            )
            snapped.validate()
            return snapped

    starts = tuple(
        snap_timestamp(
            segment.start_seconds,
            ordered_cuts,
            radius_seconds=radius_seconds,
            preference="forward" if index == 0 else "backward",
        )
        for index, segment in enumerate(result.segments)
    )
    refined: list[LineupSegment] = []
    diagnostics: list[dict[str, object]] = []
    for index, segment in enumerate(result.segments):
        start = starts[index]
        snapped_start = start.snapped_seconds
        near_audio_end = snap_timestamp(
            segment.end_seconds,
            ordered_cuts,
            radius_seconds=radius_seconds,
            preference="nearest",
        )
        if (
            near_audio_end.applied
            and near_audio_end.snapped_seconds > snapped_start + 0.25
        ):
            snapped_end = near_audio_end.snapped_seconds
            end_strategy = "near_audio_boundary"
        elif index + 1 < len(result.segments):
            next_start = starts[index + 1].snapped_seconds
            exit_candidates = tuple(
                cut
                for cut in ordered_cuts
                if snapped_start + min_graphic_duration_seconds <= cut < next_start
                and cut
                <= segment.end_seconds + max_audio_end_lead_seconds
            )
            snapped_end = max(exit_candidates) if exit_candidates else next_start
            end_strategy = "before_next_slideshow_graphic"
        else:
            eligible = tuple(
                cut
                for cut in ordered_cuts
                if snapped_start + min_graphic_duration_seconds <= cut
                <= snapped_start + max_graphic_duration_seconds
                and segment.end_seconds <= cut
                <= segment.end_seconds + max_audio_end_lead_seconds
            )
            stable_exits = tuple(
                cut
                for cut_index, cut in enumerate(ordered_cuts)
                if cut in eligible
                and cut_index > 0
                and cut - ordered_cuts[cut_index - 1]
                >= DEFAULT_SLIDESHOW_STABLE_SCENE_SECONDS
            )
            if stable_exits:
                snapped_end = min(stable_exits)
                end_strategy = "slideshow_stable_scene_exit"
            elif eligible:
                target = segment.end_seconds + 10.0
                snapped_end = min(eligible, key=lambda cut: abs(cut - target))
                end_strategy = "slideshow_audio_tail"
            else:
                fallback = snap_timestamp(
                    segment.end_seconds,
                    ordered_cuts,
                    radius_seconds=radius_seconds,
                    preference="backward",
                )
                snapped_end = fallback.snapped_seconds
                end_strategy = "near_audio_boundary"
        end = SceneSnap(
            segment.end_seconds,
            snapped_end,
            abs(snapped_end - segment.end_seconds),
            abs(snapped_end - segment.end_seconds) > 1e-6,
        )
        if snapped_end <= snapped_start + 0.25:
            refined.append(segment)
            diagnostics.append(
                {
                    "segment_id": segment.segment_id,
                    "start": start.to_dict(),
                    "end": end.to_dict(),
                    "end_strategy": "rejected_invalid_scene_range",
                }
            )
            continue
        refined.append(
            replace(
                segment,
                start_seconds=snapped_start,
                end_seconds=snapped_end,
                reason=(
                    segment.reason
                    + " PySceneDetect selected the matching slideshow graphic block."
                ),
            )
        )
        diagnostics.append(
            {
                "segment_id": segment.segment_id,
                "start": start.to_dict(),
                "end": end.to_dict(),
                "end_strategy": end_strategy,
            }
        )

    raw_response = dict(result.raw_response)
    raw_response["scene_refinement"] = {
        "method": "pyscenedetect_content_detector",
        "mode": "consecutive_slideshow_graphics",
        "radius_seconds": radius_seconds,
        "min_graphic_duration_seconds": min_graphic_duration_seconds,
        "max_graphic_duration_seconds": max_graphic_duration_seconds,
        "max_audio_end_lead_seconds": max_audio_end_lead_seconds,
        "scene_cuts_seconds": [round(cut, 3) for cut in ordered_cuts],
        "segments": diagnostics,
    }
    snapped = replace(
        result,
        segments=tuple(refined),
        raw_response=raw_response,
    )
    snapped.validate()
    return snapped


def snap_lineup_result(
    result: LineupDetectionResult,
    scene_cuts: Iterable[float],
    *,
    radius_seconds: float = DEFAULT_SCENE_SNAP_RADIUS_SECONDS,
    min_graphic_duration_seconds: float = DEFAULT_MIN_GRAPHIC_DURATION_SECONDS,
    max_graphic_duration_seconds: float = DEFAULT_MAX_GRAPHIC_DURATION_SECONDS,
    max_audio_start_offset_seconds: float = (
        DEFAULT_MAX_AUDIO_START_OFFSET_SECONDS
    ),
    max_audio_end_lead_seconds: float = DEFAULT_MAX_AUDIO_END_LEAD_SECONDS,
    max_audio_end_trail_seconds: float = DEFAULT_MAX_AUDIO_END_TRAIL_SECONDS,
    slideshow_mode: bool = False,
) -> LineupDetectionResult:
    """Snap valid lineups while preserving coarse boundaries in diagnostics."""
    cuts = tuple(scene_cuts)
    duration_values = (
        min_graphic_duration_seconds,
        max_graphic_duration_seconds,
        max_audio_start_offset_seconds,
        max_audio_end_lead_seconds,
        max_audio_end_trail_seconds,
    )
    if not all(math.isfinite(value) and value >= 0 for value in duration_values):
        raise LineupDetectionError("Graphic duration constraints must be non-negative.")
    if max_graphic_duration_seconds <= min_graphic_duration_seconds:
        raise LineupDetectionError(
            "Maximum graphic duration must exceed its minimum duration."
        )
    ordered_cuts = tuple(sorted(cuts))
    stable_scenes = tuple(
        _stable_graphic_scene(
            segment,
            ordered_cuts,
            min_graphic_duration_seconds=min_graphic_duration_seconds,
            max_graphic_duration_seconds=max_graphic_duration_seconds,
            max_audio_start_offset_seconds=max_audio_start_offset_seconds,
            max_audio_end_lead_seconds=max_audio_end_lead_seconds,
            max_audio_end_trail_seconds=max_audio_end_trail_seconds,
        )
        for segment in result.segments
    )
    auto_slideshow = _looks_like_consecutive_slideshow(
        result,
        ordered_cuts,
        stable_scenes,
        max_graphic_duration_seconds=max_graphic_duration_seconds,
    )
    if (slideshow_mode or auto_slideshow) and len(result.segments) >= 2:
        return _snap_slideshow_lineup_result(
            result,
            ordered_cuts,
            radius_seconds=radius_seconds,
            min_graphic_duration_seconds=min_graphic_duration_seconds,
            max_graphic_duration_seconds=max_graphic_duration_seconds,
            max_audio_end_lead_seconds=max_audio_end_lead_seconds,
        )
    refined: list[LineupSegment] = []
    diagnostics: list[dict[str, object]] = []
    for segment, stable_scene in zip(result.segments, stable_scenes):
        if stable_scene is not None:
            snapped_start, snapped_end = stable_scene
            start = SceneSnap(
                segment.start_seconds,
                snapped_start,
                abs(snapped_start - segment.start_seconds),
                True,
            )
            end = SceneSnap(
                segment.end_seconds,
                snapped_end,
                abs(snapped_end - segment.end_seconds),
                True,
            )
            end_strategy = "paired_graphic_block"
        else:
            start = snap_timestamp(
                segment.start_seconds,
                cuts,
                radius_seconds=radius_seconds,
                preference="forward",
            )
            snapped_start = start.snapped_seconds
            graphic_end_candidates = tuple(
                cut
                for cut in cuts
                if snapped_start + min_graphic_duration_seconds <= cut
                <= snapped_start + max_graphic_duration_seconds
                and segment.end_seconds - max_audio_end_lead_seconds <= cut
                <= segment.end_seconds + radius_seconds
            )
            if start.applied and graphic_end_candidates:
                graphic_end = min(graphic_end_candidates)
                end = SceneSnap(
                    segment.end_seconds,
                    graphic_end,
                    abs(graphic_end - segment.end_seconds),
                    True,
                )
                end_strategy = "paired_graphic_block"
            else:
                end = snap_timestamp(
                    segment.end_seconds,
                    cuts,
                    radius_seconds=radius_seconds,
                    preference="backward",
                )
                end_strategy = "near_audio_boundary"
            snapped_end = end.snapped_seconds
        if snapped_end <= snapped_start:
            snapped_start = segment.start_seconds
            snapped_end = segment.end_seconds
            start = SceneSnap(segment.start_seconds, segment.start_seconds, 0.0, False)
            end = SceneSnap(segment.end_seconds, segment.end_seconds, 0.0, False)
        refined.append(
            replace(
                segment,
                start_seconds=snapped_start,
                end_seconds=snapped_end,
                reason=(
                    segment.reason
                    + " PySceneDetect selected the matching graphic block."
                ),
            )
        )
        diagnostics.append(
            {
                "segment_id": segment.segment_id,
                "start": start.to_dict(),
                "end": end.to_dict(),
                "end_strategy": end_strategy,
            }
        )

    raw_response = dict(result.raw_response)
    raw_response["scene_refinement"] = {
        "method": "pyscenedetect_content_detector",
        "radius_seconds": radius_seconds,
        "min_graphic_duration_seconds": min_graphic_duration_seconds,
        "max_graphic_duration_seconds": max_graphic_duration_seconds,
        "max_audio_end_lead_seconds": max_audio_end_lead_seconds,
        "max_audio_end_trail_seconds": max_audio_end_trail_seconds,
        "scene_cuts_seconds": [round(cut, 3) for cut in cuts],
        "segments": diagnostics,
    }
    snapped = replace(
        result,
        segments=tuple(refined),
        raw_response=raw_response,
    )
    snapped.validate()
    return snapped
