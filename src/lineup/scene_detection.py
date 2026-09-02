"""PySceneDetect boundary snapping without a visual language model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from .schema import LineupDetectionError


DEFAULT_SCENE_THRESHOLD = 27.0
DEFAULT_SCENE_MIN_LENGTH_SECONDS = 0.5
DEFAULT_SCENE_SNAP_RADIUS_SECONDS = 4.0
DEFAULT_SOFT_TRANSITION_SCORE = 5.0


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


class SceneCut(float):
    """Timestamp carrying the detector signal that produced the cut.

    It remains a ``float`` so existing callers and persisted JSON stay fully
    compatible, while the visual refiner can distinguish broadcast cuts from
    gentle overlay-animation changes when richer data is available.
    """

    kind: str
    content_score: float | None

    def __new__(
        cls,
        seconds: float,
        *,
        kind: str = "unknown",
        content_score: float | None = None,
    ) -> SceneCut:
        instance = float.__new__(cls, seconds)
        instance.kind = kind
        instance.content_score = content_score
        return instance


def _resolve_scene_min_length_frames(
    *,
    frame_rate: float,
    min_scene_len_frames: int | None,
    min_scene_len_seconds: float,
) -> int:
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise LineupDetectionError("Video frame rate must be positive.")
    if min_scene_len_frames is not None:
        if min_scene_len_frames <= 0:
            raise LineupDetectionError(
                "Minimum scene length in frames must be positive."
            )
        return min_scene_len_frames
    if not math.isfinite(min_scene_len_seconds) or min_scene_len_seconds <= 0:
        raise LineupDetectionError(
            "Minimum scene length in seconds must be positive."
        )
    return max(1, round(min_scene_len_seconds * frame_rate))


def detect_scene_cuts(
    video_path: Path,
    *,
    start_seconds: float,
    end_seconds: float,
    threshold: float = DEFAULT_SCENE_THRESHOLD,
    min_scene_len_frames: int | None = None,
    min_scene_len_seconds: float = DEFAULT_SCENE_MIN_LENGTH_SECONDS,
) -> tuple[float, ...]:
    """Run ContentDetector only over the time range relevant to lineups."""
    source = video_path.expanduser().resolve()
    if not source.is_file():
        raise LineupDetectionError(f"Source video does not exist: {source}")
    if not all(math.isfinite(value) for value in (start_seconds, end_seconds)):
        raise LineupDetectionError("Scene detection bounds must be finite.")
    if start_seconds < 0 or end_seconds <= start_seconds:
        raise LineupDetectionError("Invalid scene detection range.")
    if threshold <= 0:
        raise LineupDetectionError(
            "Scene threshold must be positive."
        )
    if min_scene_len_frames is not None and min_scene_len_frames <= 0:
        raise LineupDetectionError("Minimum scene length in frames must be positive.")
    if not math.isfinite(min_scene_len_seconds) or min_scene_len_seconds <= 0:
        raise LineupDetectionError("Minimum scene length in seconds must be positive.")
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
        frame_rate = float(video.frame_rate)
        resolved_min_scene_len_frames = _resolve_scene_min_length_frames(
            frame_rate=frame_rate,
            min_scene_len_frames=min_scene_len_frames,
            min_scene_len_seconds=min_scene_len_seconds,
        )
        stats = StatsManager()
        manager = SceneManager(stats_manager=stats)
        manager.add_detector(
            ContentDetector(
                threshold=threshold,
                min_scene_len=resolved_min_scene_len_frames,
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
        SceneCut(scene_start.get_seconds(), kind="hard")
        for scene_start, _ in scenes[1:]
        if start_seconds <= scene_start.get_seconds() <= end_seconds
    )
    soft_cuts = _soft_transition_cuts(
        stats,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        frame_rate=frame_rate,
    )
    merged: list[float] = []
    for cut in sorted((*hard_cuts, *soft_cuts)):
        if merged and cut - merged[-1] < 0.15:
            if (
                isinstance(cut, SceneCut)
                and cut.kind == "hard"
                and (
                    not isinstance(merged[-1], SceneCut)
                    or merged[-1].kind != "hard"
                )
            ):
                merged[-1] = cut
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
        cuts.append(
            SceneCut(
                frame_number / frame_rate,
                kind="soft_transition",
                content_score=score,
            )
        )
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
        gentle_cuts.append(
            SceneCut(
                frame_number / frame_rate,
                kind="gentle_overlay",
                content_score=score,
            )
        )
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
    nearest = float(min(pool, key=lambda cut: (abs(cut - timestamp), cut)))
    distance = abs(nearest - timestamp)
    return SceneSnap(timestamp, nearest, distance, True)
