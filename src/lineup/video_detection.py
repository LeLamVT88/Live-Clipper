"""Transcript-free lineup detection from scene representatives and tiny OCR."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

from .scene_detection import detect_scene_cuts
from .schema import LineupDetectionError, LineupDetectionResult, LineupSegment
from .visual_scan import (
    DEFAULT_OCR_PYTHON,
    MIN_CONFIDENT_LINEUP_DURATION_SECONDS,
    OCRRunner,
    SceneDetector,
    VisualEvent,
    VisualSearchTask,
    build_global_task,
    build_visual_units,
    events_from_worker,
    run_ocr_worker,
    snap_visual_event,
    visual_team_hint,
)


DEFAULT_COARSE_UNIT_SECONDS = 3.0
DEFAULT_DENSE_UNIT_SECONDS = 0.5
DEFAULT_DENSE_PADDING_SECONDS = 12.0
DEFAULT_SAFETY_PADDING_BEFORE_SECONDS = 4.0
DEFAULT_SAFETY_PADDING_AFTER_SECONDS = 6.0
DEFAULT_MAX_COARSE_CANDIDATES = 6
DEFAULT_VISUAL_MODEL = "PP-OCRv6_tiny_det+PP-OCRv6_tiny_rec"


def _validate_positive(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise LineupDetectionError(f"{name} must be finite and positive.")
    return number


def _overlaps(first: VisualEvent, second: VisualEvent) -> bool:
    return (
        first.start_seconds < second.end_seconds
        and second.start_seconds < first.end_seconds
    )


def _choose_dense_event(
    coarse: VisualEvent,
    candidates: Sequence[VisualEvent],
) -> VisualEvent | None:
    overlapping = tuple(
        event for event in candidates if _overlaps(coarse, event)
    )
    if not overlapping:
        return None
    chosen = min(
        overlapping,
        key=lambda event: (
            abs(event.center_seconds - coarse.center_seconds),
            -event.confidence,
        ),
    )
    texts = tuple(dict.fromkeys((*coarse.texts, *chosen.texts)))
    samples = tuple(sorted(set((*coarse.sample_seconds, *chosen.sample_seconds))))
    return replace(
        chosen,
        texts=texts,
        sample_seconds=samples,
    )


def _select_events(
    events: Sequence[VisualEvent],
    *,
    limit: int,
) -> tuple[VisualEvent, ...]:
    selected: list[VisualEvent] = []
    for event in sorted(
        events,
        key=lambda item: (-item.confidence, item.start_seconds),
    ):
        if not any(_overlaps(event, existing) for existing in selected):
            selected.append(event)
        if len(selected) >= limit:
            break
    return tuple(sorted(selected, key=lambda item: item.start_seconds))


def _pad_events_without_overlap(
    events: Sequence[VisualEvent],
    *,
    scan_start_seconds: float,
    scan_end_seconds: float,
    before_seconds: float,
    after_seconds: float,
) -> tuple[VisualEvent, ...]:
    if not events:
        return ()
    padded = [
        replace(
            event,
            start_seconds=max(
                scan_start_seconds,
                event.start_seconds - before_seconds,
            ),
            end_seconds=min(
                scan_end_seconds,
                event.end_seconds + after_seconds,
            ),
        )
        for event in events
    ]
    for index in range(len(padded) - 1):
        left = padded[index]
        right = padded[index + 1]
        if left.end_seconds <= right.start_seconds:
            continue
        raw_left = events[index]
        raw_right = events[index + 1]
        boundary = (raw_left.end_seconds + raw_right.start_seconds) / 2
        padded[index] = replace(left, end_seconds=boundary)
        padded[index + 1] = replace(right, start_seconds=boundary)
    return tuple(padded)


def _dense_tasks(
    events: Sequence[VisualEvent],
    *,
    scene_cuts: Sequence[float],
    scan_start_seconds: float,
    scan_end_seconds: float,
    padding_seconds: float,
    unit_seconds: float,
) -> tuple[VisualSearchTask, ...]:
    tasks: list[VisualSearchTask] = []
    for index, event in enumerate(events, start=1):
        start = max(scan_start_seconds, event.start_seconds - padding_seconds)
        end = min(scan_end_seconds, event.end_seconds + padding_seconds)
        if end <= start:
            continue
        tasks.append(
            VisualSearchTask(
                task_id=f"dense_{index:03d}",
                mode="dense",
                center_seconds=event.center_seconds,
                max_events=3,
                units=build_visual_units(
                    start_seconds=start,
                    end_seconds=end,
                    scene_cuts=scene_cuts,
                    max_unit_seconds=unit_seconds,
                ),
            )
        )
    return tuple(tasks)


def _result_from_events(
    events: Sequence[VisualEvent],
    *,
    evidence_events: Sequence[VisualEvent],
    expected_count: int | None,
    diagnostics: Mapping[str, object],
    dense_misses: Sequence[str],
) -> LineupDetectionResult:
    review_reasons: set[str] = set()
    if expected_count is not None and len(events) != expected_count:
        review_reasons.add("visual_lineup_count_does_not_match_expected")
    if dense_misses:
        review_reasons.add("dense_ocr_did_not_reconfirm_all_candidates")
    if int(diagnostics.get("refined_candidate_count", 0)) > len(events):
        review_reasons.add("more_visual_candidates_than_output_capacity")

    segments: list[LineupSegment] = []
    used_team_names: set[str] = set()
    for index, (event, evidence) in enumerate(
        zip(events, evidence_events, strict=True), start=1
    ):
        team_name = visual_team_hint(evidence)
        team_key = team_name.casefold() if team_name else ""
        if not team_name or team_key in used_team_names:
            team_name = f"Visual lineup {index}"
            team_key = team_name.casefold()
            review_reasons.add("visual_lineup_team_name_unreadable")
        used_team_names.add(team_key)
        if evidence.end_seconds - evidence.start_seconds < (
            MIN_CONFIDENT_LINEUP_DURATION_SECONDS
        ):
            review_reasons.add("visual_lineup_boundary_may_be_incomplete")
        segments.append(
            LineupSegment(
                segment_id=f"lineup_{index:03d}",
                team_name=team_name,
                start_seconds=event.start_seconds,
                end_seconds=event.end_seconds,
                confidence=evidence.confidence,
                evidence_segment_ids=(f"visual_ocr_{index:03d}",),
                start_anchor_text="visual OCR lineup onset",
                end_anchor_text="visual OCR lineup exit",
                reason=(
                    "Scene-representative OCR found a lineup graphic; dense "
                    "visual scanning refined both boundaries and conservative "
                    "padding was added."
                ),
                evidence_start_seconds=evidence.start_seconds,
                evidence_end_seconds=evidence.end_seconds,
            )
        )

    if not segments:
        status = "empty"
    elif expected_count is None:
        status = "unconstrained"
    elif len(segments) == expected_count:
        status = "complete"
    else:
        status = "incomplete"
    raw_response = {
        "lineup_segments": [event.to_dict() for event in events],
        "visual_detection": dict(diagnostics),
        "_review_reasons": sorted(review_reasons),
    }
    result = LineupDetectionResult(
        model=DEFAULT_VISUAL_MODEL,
        segments=tuple(segments),
        raw_response=raw_response,
        status=status,
    )
    result.validate()
    return result


def detect_lineups_from_video(
    video_path: Path,
    *,
    scan_start_seconds: float,
    scan_end_seconds: float,
    expected_count: int | None,
    scene_threshold: float,
    scene_min_length_frames: int | None,
    scene_min_length_seconds: float,
    scene_snap_radius_seconds: float,
    coarse_unit_seconds: float = DEFAULT_COARSE_UNIT_SECONDS,
    dense_unit_seconds: float = DEFAULT_DENSE_UNIT_SECONDS,
    dense_padding_seconds: float = DEFAULT_DENSE_PADDING_SECONDS,
    safety_padding_before_seconds: float = (
        DEFAULT_SAFETY_PADDING_BEFORE_SECONDS
    ),
    safety_padding_after_seconds: float = DEFAULT_SAFETY_PADDING_AFTER_SECONDS,
    max_coarse_candidates: int = DEFAULT_MAX_COARSE_CANDIDATES,
    ocr_python: Path = DEFAULT_OCR_PYTHON,
    scene_detector: SceneDetector = detect_scene_cuts,
    ocr_runner: OCRRunner = run_ocr_worker,
) -> LineupDetectionResult:
    """Detect lineup intervals without ASR or transcript evidence.

    The coarse pass runs text detection/recognition on representatives from
    every scene (long scenes are split into bounded units).  A 0.5-second
    second pass around each candidate refines the boundaries.  Coarse evidence
    is retained when the dense pass fails so recall is never reduced by the
    refinement stage.
    """

    start = float(scan_start_seconds)
    end = float(scan_end_seconds)
    if not all(math.isfinite(value) for value in (start, end)):
        raise LineupDetectionError("Visual scan bounds must be finite.")
    if start < 0 or end <= start:
        raise LineupDetectionError("Invalid visual scan range.")
    if expected_count not in {None, 1, 2}:
        raise LineupDetectionError("Expected lineup count must be 1, 2, or auto.")
    if max_coarse_candidates < 1:
        raise LineupDetectionError("Maximum coarse candidates must be positive.")
    coarse_unit = _validate_positive("coarse_unit_seconds", coarse_unit_seconds)
    dense_unit = _validate_positive("dense_unit_seconds", dense_unit_seconds)
    dense_padding = _validate_positive(
        "dense_padding_seconds", dense_padding_seconds
    )
    before_padding = float(safety_padding_before_seconds)
    after_padding = float(safety_padding_after_seconds)
    if not all(
        math.isfinite(value) and value >= 0
        for value in (before_padding, after_padding)
    ):
        raise LineupDetectionError("Safety padding must be finite and non-negative.")

    detector_options = {
        "threshold": scene_threshold,
        "min_scene_len_frames": scene_min_length_frames,
        "min_scene_len_seconds": scene_min_length_seconds,
    }
    scene_cuts = scene_detector(
        video_path,
        start_seconds=start,
        end_seconds=end,
        **detector_options,
    )
    coarse_task = build_global_task(
        scene_cuts=scene_cuts,
        scan_start_seconds=start,
        scan_end_seconds=end,
        max_events=max_coarse_candidates,
        max_unit_seconds=coarse_unit,
    )
    coarse_payload = ocr_runner(
        video_path,
        (coarse_task,),
        python_path=ocr_python,
        max_recognition_frames=max(1, len(coarse_task.units)),
    )
    coarse_events = events_from_worker(coarse_payload).get(coarse_task.task_id, ())

    dense_tasks = _dense_tasks(
        coarse_events,
        scene_cuts=scene_cuts,
        scan_start_seconds=start,
        scan_end_seconds=end,
        padding_seconds=dense_padding,
        unit_seconds=dense_unit,
    )
    dense_payload: dict[str, object] = {"tasks": []}
    dense_events_by_task: dict[str, tuple[VisualEvent, ...]] = {}
    if dense_tasks:
        dense_payload = ocr_runner(
            video_path,
            dense_tasks,
            python_path=ocr_python,
            max_recognition_frames=max(len(task.units) for task in dense_tasks),
        )
        dense_events_by_task = events_from_worker(dense_payload)

    refined_events: list[VisualEvent] = []
    dense_misses: list[str] = []
    for coarse, task in zip(coarse_events, dense_tasks, strict=True):
        refined = _choose_dense_event(
            coarse,
            dense_events_by_task.get(task.task_id, ()),
        )
        if refined is None:
            refined = coarse
            dense_misses.append(task.task_id)
        refined_events.append(
            snap_visual_event(
                refined,
                scene_cuts,
                radius_seconds=scene_snap_radius_seconds,
            )
        )

    output_limit = expected_count if expected_count is not None else 2
    selected_evidence = _select_events(refined_events, limit=output_limit)
    padded_events = _pad_events_without_overlap(
        selected_evidence,
        scan_start_seconds=start,
        scan_end_seconds=end,
        before_seconds=before_padding,
        after_seconds=after_padding,
    )
    diagnostics = {
        "method": "scene_representative_ocr_then_dense_visual_scan",
        "scan_start_seconds": round(start, 3),
        "scan_end_seconds": round(end, 3),
        "coarse_unit_max_seconds": coarse_unit,
        "dense_unit_max_seconds": dense_unit,
        "dense_padding_seconds": dense_padding,
        "safety_padding_before_seconds": before_padding,
        "safety_padding_after_seconds": after_padding,
        "scene_threshold": scene_threshold,
        "scene_min_length_frames": scene_min_length_frames,
        "scene_min_length_seconds": scene_min_length_seconds,
        "scene_snap_radius_seconds": scene_snap_radius_seconds,
        "scene_cuts_seconds": [round(float(cut), 3) for cut in scene_cuts],
        "coarse_candidate_count": len(coarse_events),
        "refined_candidate_count": len(refined_events),
        "selected_candidate_count": len(selected_evidence),
        "dense_misses": dense_misses,
        "coarse_ocr": coarse_payload,
        "dense_ocr": dense_payload,
        "refined_events": [event.to_dict() for event in refined_events],
        "selected_evidence_events": [
            event.to_dict() for event in selected_evidence
        ],
    }
    return _result_from_events(
        padded_events,
        evidence_events=selected_evidence,
        expected_count=expected_count,
        diagnostics=diagnostics,
        dense_misses=dense_misses,
    )
