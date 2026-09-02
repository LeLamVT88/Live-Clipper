"""Plan scene-level OCR searches and merge visual lineup evidence."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .scene_detection import detect_scene_cuts
from .schema import LineupDetectionError, LineupDetectionResult, LineupSegment
from .utils import PROJECT_ROOT


MAX_SCAN_SECONDS = 600.0
DEFAULT_UNIT_SECONDS = 3.0
DEFAULT_LOCAL_PADDING_SECONDS = 12.0
DEFAULT_LOCAL_RECOGNITION_FRAMES = 24
DEFAULT_MAX_RECOGNITION_FRAMES = 40
MIN_CONFIDENT_LINEUP_DURATION_SECONDS = 8.0
MIN_UNMATCHED_LOCAL_CONFIDENCE = 0.7
DEFAULT_OCR_PYTHON = PROJECT_ROOT / ".venv-ocr" / "bin" / "python"

TEAM_LABEL_EXCLUSIONS = frozenset(
    {
        "COACH",
        "FORMATION",
        "FIFA",
        "GK",
        "LINEUP",
        "STARTING XI",
        "SUBSTITUTES",
    }
)

SceneDetector = Callable[..., tuple[float, ...]]
OCRRunner = Callable[..., dict[str, object]]


def _value_or(value: float | None, fallback: float) -> float:
    return fallback if value is None else value


@dataclass(frozen=True)
class VisualUnit:
    """One visually stable interval represented by its middle frame."""

    start_seconds: float
    end_seconds: float
    scene_index: int

    @property
    def center_seconds(self) -> float:
        return (self.start_seconds + self.end_seconds) / 2

    def to_dict(self) -> dict[str, float | int]:
        return {
            "start_seconds": round(self.start_seconds, 3),
            "end_seconds": round(self.end_seconds, 3),
            "sample_seconds": round(self.center_seconds, 3),
            "scene_index": self.scene_index,
        }


@dataclass(frozen=True)
class VisualSearchTask:
    task_id: str
    mode: str
    center_seconds: float
    max_events: int
    units: tuple[VisualUnit, ...]

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "task_id": self.task_id,
            "mode": self.mode,
            "center_seconds": round(self.center_seconds, 3),
            "max_events": self.max_events,
            "units": [unit.to_dict() for unit in self.units],
        }
        return payload


@dataclass(frozen=True)
class VisualEvent:
    start_seconds: float
    end_seconds: float
    confidence: float
    task_id: str
    sample_seconds: tuple[float, ...] = ()
    texts: tuple[str, ...] = ()

    @property
    def center_seconds(self) -> float:
        return (self.start_seconds + self.end_seconds) / 2

    def to_dict(self) -> dict[str, object]:
        return {
            "start_seconds": round(self.start_seconds, 3),
            "end_seconds": round(self.end_seconds, 3),
            "confidence": round(self.confidence, 4),
            "task_id": self.task_id,
            "sample_seconds": [round(value, 3) for value in self.sample_seconds],
            "texts": list(self.texts),
        }


def build_visual_units(
    *,
    start_seconds: float,
    end_seconds: float,
    scene_cuts: Iterable[float],
    max_unit_seconds: float = DEFAULT_UNIT_SECONDS,
) -> tuple[VisualUnit, ...]:
    """Split physical scenes only when they are too long for overlay changes."""
    if not all(math.isfinite(value) for value in (start_seconds, end_seconds)):
        raise LineupDetectionError("Visual search bounds must be finite.")
    if start_seconds < 0 or end_seconds <= start_seconds:
        raise LineupDetectionError("Invalid visual search range.")
    if not math.isfinite(max_unit_seconds) or max_unit_seconds <= 0:
        raise LineupDetectionError("Visual unit duration must be positive.")

    boundaries = [start_seconds]
    boundaries.extend(
        sorted(
            {
                float(cut)
                for cut in scene_cuts
                if math.isfinite(float(cut))
                and start_seconds < float(cut) < end_seconds
            }
        )
    )
    boundaries.append(end_seconds)

    units: list[VisualUnit] = []
    for scene_index, (scene_start, scene_end) in enumerate(
        zip(boundaries, boundaries[1:])
    ):
        part_count = max(1, math.ceil((scene_end - scene_start) / max_unit_seconds))
        part_duration = (scene_end - scene_start) / part_count
        for part_index in range(part_count):
            unit_start = scene_start + part_index * part_duration
            unit_end = (
                scene_end
                if part_index == part_count - 1
                else scene_start + (part_index + 1) * part_duration
            )
            units.append(VisualUnit(unit_start, unit_end, scene_index))
    return tuple(units)


def build_local_tasks(
    segments: Sequence[LineupSegment],
    *,
    scene_cuts: Iterable[float],
    scan_start_seconds: float,
    scan_end_seconds: float,
    padding_seconds: float = DEFAULT_LOCAL_PADDING_SECONDS,
) -> tuple[VisualSearchTask, ...]:
    """Create one middle-out OCR task around each coarse Qwen interval."""
    cuts = tuple(scene_cuts)
    tasks: list[VisualSearchTask] = []
    for segment in segments:
        start, end = _local_search_bounds(
            segment,
            scan_start_seconds=scan_start_seconds,
            scan_end_seconds=scan_end_seconds,
            padding_seconds=padding_seconds,
        )
        if end <= start:
            continue
        tasks.append(
            VisualSearchTask(
                task_id=segment.segment_id,
                mode="local",
                center_seconds=(segment.start_seconds + segment.end_seconds) / 2,
                max_events=1,
                units=build_visual_units(
                    start_seconds=start,
                    end_seconds=end,
                    scene_cuts=cuts,
                ),
            )
        )
    return tuple(tasks)


def _local_search_bounds(
    segment: LineupSegment,
    *,
    scan_start_seconds: float,
    scan_end_seconds: float,
    padding_seconds: float = DEFAULT_LOCAL_PADDING_SECONDS,
) -> tuple[float, float]:
    start = max(
        scan_start_seconds,
        min(
            segment.start_seconds,
            _value_or(segment.evidence_start_seconds, segment.start_seconds),
        )
        - padding_seconds,
    )
    end = min(
        scan_end_seconds,
        max(
            segment.end_seconds,
            _value_or(segment.evidence_end_seconds, segment.end_seconds),
        )
        + padding_seconds,
    )
    return start, end


def _merge_search_ranges(
    ranges: Iterable[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(ranges):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return tuple(merged)


def build_global_task(
    *,
    scene_cuts: Iterable[float],
    scan_start_seconds: float,
    scan_end_seconds: float,
    max_events: int = 2,
) -> VisualSearchTask:
    """Build the rare 600-second fallback using one representative per unit."""
    return VisualSearchTask(
        task_id="global_fallback",
        mode="global_fallback",
        center_seconds=(scan_start_seconds + scan_end_seconds) / 2,
        max_events=max_events,
        units=build_visual_units(
            start_seconds=scan_start_seconds,
            end_seconds=scan_end_seconds,
            scene_cuts=scene_cuts,
        ),
    )


def run_ocr_worker(
    video_path: Path,
    tasks: Sequence[VisualSearchTask],
    *,
    python_path: Path = DEFAULT_OCR_PYTHON,
    max_recognition_frames: int = DEFAULT_MAX_RECOGNITION_FRAMES,
) -> dict[str, object]:
    """Run lightweight PaddleOCR in its dedicated Python environment."""
    if not tasks:
        return {"tasks": [], "model_loaded": False}
    executable = python_path.expanduser()
    if not executable.is_absolute():
        executable = PROJECT_ROOT / executable
    if not executable.is_file():
        raise LineupDetectionError(
            "OCR Python does not exist: "
            f"{executable}. Create .venv-ocr or pass --ocr-python."
        )
    worker_path = Path(__file__).with_name("ocr_worker.py")
    request = {
        "video_path": str(video_path.resolve()),
        "max_recognition_frames": max_recognition_frames,
        "tasks": [task.to_dict() for task in tasks],
    }
    with tempfile.TemporaryDirectory(prefix="lineup_ocr_") as directory:
        request_path = Path(directory) / "request.json"
        output_path = Path(directory) / "response.json"
        request_path.write_text(
            json.dumps(request, ensure_ascii=False), encoding="utf-8"
        )
        environment = dict(os.environ)
        environment.setdefault(
            "PADDLE_PDX_CACHE_HOME", str(PROJECT_ROOT / ".cache" / "paddlex")
        )
        environment.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        completed = subprocess.run(
            [
                str(executable),
                str(worker_path),
                "--request",
                str(request_path),
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
        if completed.returncode != 0 or not output_path.is_file():
            detail = (completed.stderr or completed.stdout).strip()[-1500:]
            raise LineupDetectionError(
                f"Lineup OCR worker failed ({completed.returncode}): {detail}"
            )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise LineupDetectionError("Lineup OCR worker returned invalid JSON.")
    return payload


def events_from_worker(
    payload: Mapping[str, object],
) -> dict[str, tuple[VisualEvent, ...]]:
    """Validate the compact worker response at the process boundary."""
    parsed: dict[str, tuple[VisualEvent, ...]] = {}
    raw_tasks = payload.get("tasks", [])
    if not isinstance(raw_tasks, list):
        raise LineupDetectionError("Lineup OCR tasks must be an array.")
    for raw_task in raw_tasks:
        if not isinstance(raw_task, dict):
            raise LineupDetectionError("Invalid lineup OCR task result.")
        task_id = str(raw_task.get("task_id", "")).strip()
        raw_events = raw_task.get("events", [])
        if not task_id or not isinstance(raw_events, list):
            raise LineupDetectionError("Invalid lineup OCR task result.")
        events: list[VisualEvent] = []
        for raw_event in raw_events:
            if not isinstance(raw_event, dict):
                raise LineupDetectionError("Invalid lineup OCR event.")
            try:
                event = VisualEvent(
                    start_seconds=float(raw_event["start_seconds"]),
                    end_seconds=float(raw_event["end_seconds"]),
                    confidence=float(raw_event["confidence"]),
                    task_id=task_id,
                    sample_seconds=tuple(
                        float(value)
                        for value in raw_event.get("sample_seconds", [])
                    ),
                    texts=tuple(
                        str(value) for value in raw_event.get("texts", [])
                    ),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise LineupDetectionError("Invalid lineup OCR event values.") from exc
            if (
                event.start_seconds < 0
                or event.end_seconds <= event.start_seconds
                or not 0 <= event.confidence <= 1
            ):
                raise LineupDetectionError("Invalid lineup OCR event range.")
            events.append(event)
        parsed[task_id] = tuple(events)
    return parsed


def snap_visual_event(
    event: VisualEvent,
    scene_cuts: Iterable[float],
    *,
    radius_seconds: float,
) -> VisualEvent:
    """Expand OCR bounds to enclosing cuts without trimming the graphic.

    OCR identifies visual evidence *inside* a lineup.  Snapping to the nearest
    cut can therefore choose a player-slide transition and shorten the clip.
    Only an earlier start or a later end is safe here; when no cut exists in
    that direction, keep the OCR boundary.
    """
    cuts = tuple(
        sorted(
            float(cut)
            for cut in scene_cuts
            if math.isfinite(float(cut)) and float(cut) >= 0
        )
    )
    earlier_starts = tuple(
        cut
        for cut in cuts
        if 1e-6 < event.start_seconds - cut <= radius_seconds
    )
    later_ends = tuple(
        cut
        for cut in cuts
        if -1e-3 <= cut - event.end_seconds <= radius_seconds
    )
    start = max(earlier_starts, default=event.start_seconds)
    end = min(later_ends, default=event.end_seconds)
    if end <= start + 0.25:
        return event
    return replace(event, start_seconds=start, end_seconds=end)


def _non_overlapping_count(events: Iterable[VisualEvent]) -> int:
    end_seconds = -1.0
    count = 0
    for event in sorted(events, key=lambda item: item.start_seconds):
        if event.start_seconds >= end_seconds:
            count += 1
            end_seconds = event.end_seconds
    return count


def _normalized_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).casefold()
    return " ".join(
        "".join(
            character
            for character in normalized
            if not unicodedata.combining(character)
            and (character.isalnum() or character.isspace())
        ).split()
    )


def _events_overlap(first: VisualEvent, second: VisualEvent) -> bool:
    return (
        first.start_seconds < second.end_seconds
        and second.start_seconds < first.end_seconds
    )


def _event_matches_team(event: VisualEvent, team_name: str) -> bool:
    team = _normalized_text(team_name)
    return bool(team) and any(
        team in _normalized_text(text) for text in event.texts
    )


def _credible_local_events(
    events: Sequence[VisualEvent],
    team_name: str,
) -> tuple[VisualEvent, ...]:
    matching = tuple(
        event for event in events if _event_matches_team(event, team_name)
    )
    if matching:
        return matching
    return tuple(
        event
        for event in events
        if event.confidence >= MIN_UNMATCHED_LOCAL_CONFIDENCE
    )


def _visual_team_hint(event: VisualEvent) -> str | None:
    """Read only a likely team title, not player-level OCR details."""
    for raw_text in event.texts[:4]:
        text = " ".join(raw_text.strip().split())
        normalized = _normalized_text(text).upper()
        letters = "".join(character for character in text if character.isalpha())
        if (
            3 <= len(text) <= 40
            and letters
            and text == text.upper()
            and normalized not in TEAM_LABEL_EXCLUSIONS
            and all(character.isalpha() or character in " -.'" for character in text)
        ):
            return text
    return None


def merge_visual_evidence(
    result: LineupDetectionResult,
    *,
    local_events: Mapping[str, Sequence[VisualEvent]],
    fallback_events: Sequence[VisualEvent],
    expected_count: int | None,
    diagnostics: Mapping[str, object],
) -> LineupDetectionResult:
    """Prefer local OCR, then use global events only for unresolved candidates."""
    assigned: list[tuple[LineupSegment | None, VisualEvent]] = []
    unresolved: list[LineupSegment] = []
    for segment in result.segments:
        candidates = tuple(local_events.get(segment.segment_id, ()))
        if candidates:
            center = (segment.start_seconds + segment.end_seconds) / 2
            matching = tuple(
                event
                for event in candidates
                if _event_matches_team(event, segment.team_name)
            )
            event = min(
                matching or candidates,
                key=lambda item: abs(item.center_seconds - center),
            )
            assigned.append((segment, event))
        else:
            unresolved.append(segment)

    used_fallback = {
        index
        for index, event in enumerate(fallback_events)
        if any(_events_overlap(event, local_event) for _, local_event in assigned)
    }
    for segment in unresolved:
        available = [
            (index, event)
            for index, event in enumerate(fallback_events)
            if index not in used_fallback
        ]
        if not available:
            break
        center = (segment.start_seconds + segment.end_seconds) / 2
        matching = [
            item
            for item in available
            if _event_matches_team(item[1], segment.team_name)
        ]
        index, event = min(
            matching or available,
            key=lambda item: abs(item[1].center_seconds - center),
        )
        used_fallback.add(index)
        assigned.append((segment, event))

    desired_count = expected_count if expected_count is not None else 2
    for index, event in enumerate(fallback_events):
        if index in used_fallback:
            continue
        assigned.append((None, event))
        used_fallback.add(index)

    final_pairs: list[tuple[LineupSegment | None, VisualEvent]] = []
    for pair in sorted(
        assigned,
        key=lambda item: (
            item[0] is None,
            -item[1].confidence,
            item[1].start_seconds,
        ),
    ):
        event = pair[1]
        overlaps = any(
            _events_overlap(event, selected) for _, selected in final_pairs
        )
        if not overlaps:
            final_pairs.append(pair)
        if len(final_pairs) >= desired_count:
            break

    refined: list[LineupSegment] = []
    for index, (segment, event) in enumerate(
        sorted(final_pairs, key=lambda item: item[1].start_seconds), start=1
    ):
        if segment is None:
            team_hint = _visual_team_hint(event)
            refined.append(
                LineupSegment(
                    segment_id=f"lineup_{index:03d}",
                    team_name=team_hint or f"Visual lineup {index}",
                    start_seconds=event.start_seconds,
                    end_seconds=event.end_seconds,
                    confidence=event.confidence,
                    evidence_segment_ids=(f"visual_ocr_{index:03d}",),
                    start_anchor_text="visual OCR lineup onset",
                    end_anchor_text="visual OCR lineup exit",
                    reason=(
                        "Tiny OCR found a lineup graphic without matching "
                        "commentary, read its team title when available, and "
                        "PySceneDetect aligned its boundaries."
                    ),
                )
            )
            continue
        refined.append(
            replace(
                segment,
                segment_id=f"lineup_{index:03d}",
                start_seconds=event.start_seconds,
                end_seconds=event.end_seconds,
                confidence=max(segment.confidence, event.confidence),
                reason=(
                    segment.reason
                    + " Tiny OCR confirmed the graphic and PySceneDetect "
                    "aligned its boundaries."
                ),
            )
        )

    raw_review_reasons = result.raw_response.get("_review_reasons", [])
    review_reasons = set(
        raw_review_reasons if isinstance(raw_review_reasons, list) else []
    )
    if result.segments and len(refined) < len(result.segments):
        review_reasons.add("ocr_could_not_confirm_all_qwen_candidates")
    if expected_count is not None and len(refined) != expected_count:
        review_reasons.add("visual_lineup_count_does_not_match_expected")
    if any(segment.team_name.startswith("Visual lineup ") for segment in refined):
        review_reasons.add("visual_lineup_has_no_transcript_team_name")
    if any(
        segment.evidence_segment_ids
        and segment.evidence_segment_ids[0].startswith("visual_ocr_")
        for segment in refined
    ):
        review_reasons.add("visual_lineup_has_no_transcript_evidence")
    if any(
        segment.end_seconds - segment.start_seconds
        < MIN_CONFIDENT_LINEUP_DURATION_SECONDS
        for segment in refined
    ):
        review_reasons.add("visual_lineup_boundary_may_be_incomplete")

    if not refined:
        status = "empty"
    elif expected_count is None:
        status = "unconstrained"
    elif len(refined) == expected_count:
        status = "complete"
    else:
        status = "incomplete"
    raw_response = dict(result.raw_response)
    raw_response["visual_refinement"] = dict(diagnostics)
    raw_response["visual_refinement"]["events"] = [
        event.to_dict() for _, event in final_pairs
    ]
    raw_response["_review_reasons"] = sorted(review_reasons)
    merged = replace(
        result,
        segments=tuple(refined),
        raw_response=raw_response,
        status=status,
    )
    merged.validate()
    return merged


def refine_with_visual_ocr(
    result: LineupDetectionResult,
    *,
    video_path: Path,
    scan_end_seconds: float,
    expected_count: int | None,
    scene_threshold: float,
    scene_min_length_frames: int | None,
    scene_min_length_seconds: float,
    scene_snap_radius_seconds: float,
    ocr_python: Path = DEFAULT_OCR_PYTHON,
    scene_detector: SceneDetector = detect_scene_cuts,
    ocr_runner: OCRRunner = run_ocr_worker,
) -> LineupDetectionResult:
    """Run local middle-out OCR, with a sparse 600-second fallback if needed."""
    scan_end = min(float(scan_end_seconds), MAX_SCAN_SECONDS)
    if not math.isfinite(scan_end) or scan_end <= 0:
        raise LineupDetectionError("Visual scan duration must be positive.")

    detector_options = {
        "threshold": scene_threshold,
        "min_scene_len_frames": scene_min_length_frames,
        "min_scene_len_seconds": scene_min_length_seconds,
    }
    local_cuts: tuple[float, ...] = ()
    local_tasks: tuple[VisualSearchTask, ...] = ()
    local_payload: dict[str, object] = {"tasks": []}
    local_events: dict[str, tuple[VisualEvent, ...]] = {}
    local_ranges: tuple[tuple[float, float], ...] = ()
    if result.segments:
        local_ranges = _merge_search_ranges(
            _local_search_bounds(
                segment,
                scan_start_seconds=0.0,
                scan_end_seconds=scan_end,
            )
            for segment in result.segments
        )
        if local_ranges:
            local_cuts = tuple(
                sorted(
                    {
                        cut
                        for local_start, local_end in local_ranges
                        for cut in scene_detector(
                            video_path,
                            start_seconds=local_start,
                            end_seconds=local_end,
                            **detector_options,
                        )
                    }
                )
            )
            local_tasks = build_local_tasks(
                result.segments,
                scene_cuts=local_cuts,
                scan_start_seconds=0.0,
                scan_end_seconds=scan_end,
            )
            local_payload = ocr_runner(
                video_path,
                local_tasks,
                python_path=ocr_python,
                max_recognition_frames=DEFAULT_LOCAL_RECOGNITION_FRAMES,
            )
            local_events = events_from_worker(local_payload)
            local_events = {
                task_id: tuple(
                    snap_visual_event(
                        event,
                        local_cuts,
                        radius_seconds=scene_snap_radius_seconds,
                    )
                    for event in events
                )
                for task_id, events in local_events.items()
            }
            segments_by_id = {
                segment.segment_id: segment for segment in result.segments
            }
            local_events = {
                task_id: _credible_local_events(
                    events,
                    segments_by_id[task_id].team_name,
                )
                for task_id, events in local_events.items()
                if task_id in segments_by_id
            }

    confirmed_local_count = _non_overlapping_count(
        events[0] for events in local_events.values() if events
    )
    target_count = (
        expected_count
        if expected_count is not None
        else max(1, len(result.segments))
    )
    fallback_used = confirmed_local_count < target_count
    fallback_cuts: tuple[float, ...] = ()
    fallback_payload: dict[str, object] = {"tasks": []}
    fallback_events: tuple[VisualEvent, ...] = ()
    if fallback_used:
        fallback_cuts = scene_detector(
            video_path,
            start_seconds=0.0,
            end_seconds=scan_end,
            **detector_options,
        )
        fallback_task = build_global_task(
            scene_cuts=fallback_cuts,
            scan_start_seconds=0.0,
            scan_end_seconds=scan_end,
            max_events=expected_count or 2,
        )
        fallback_payload = ocr_runner(
            video_path,
            (fallback_task,),
            python_path=ocr_python,
        )
        fallback_events = tuple(
            snap_visual_event(
                event,
                fallback_cuts,
                radius_seconds=scene_snap_radius_seconds,
            )
            for event in events_from_worker(fallback_payload).get(
                fallback_task.task_id, ()
            )
        )

    diagnostics = {
        "method": "scene_representative_tiny_ocr_middle_out",
        "scan_limit_seconds": MAX_SCAN_SECONDS,
        "unit_max_seconds": DEFAULT_UNIT_SECONDS,
        "local_recognition_frame_limit": DEFAULT_LOCAL_RECOGNITION_FRAMES,
        "fallback_recognition_frame_limit": DEFAULT_MAX_RECOGNITION_FRAMES,
        "scene_threshold": scene_threshold,
        "scene_min_length_frames": scene_min_length_frames,
        "scene_min_length_seconds": scene_min_length_seconds,
        "scene_snap_radius_seconds": scene_snap_radius_seconds,
        "local_scene_ranges_seconds": [
            [round(start, 3), round(end, 3)] for start, end in local_ranges
        ],
        "local_scene_cuts_seconds": [round(cut, 3) for cut in local_cuts],
        "local_ocr": local_payload,
        "fallback_used": fallback_used,
        "fallback_reason": (
            "missing_or_unconfirmed_qwen_candidate" if fallback_used else None
        ),
        "fallback_scene_cuts_seconds": [
            round(cut, 3) for cut in fallback_cuts
        ],
        "fallback_ocr": fallback_payload,
    }
    return merge_visual_evidence(
        result,
        local_events=local_events,
        fallback_events=fallback_events,
        expected_count=expected_count,
        diagnostics=diagnostics,
    )
