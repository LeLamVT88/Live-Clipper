"""Run and evaluate one adaptive OCR attempt."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .config import PipelineConfig
from .ocr_engine import run_ocr
from .resolver import (
    LineupResolutionError,
    QualityResult,
    evaluate_segment_quality,
    resolve_all_lineups,
)
from .selector import SegmentSelection


SegmentKey = tuple[str, int]
ATTEMPT_COLUMNS = [
    "video",
    "segment_index",
    "attempt",
    "tier",
    "frame_count",
    "new_ocr_frame_count",
    "detection_count",
    "resolver_status",
    "resolved_players",
    "quality_pass",
    "quality_message",
]


@dataclass
class AttemptOutcome:
    tier: str
    frame_records: dict[SegmentKey, list[dict[str, object]]]
    detections: dict[SegmentKey, list[dict[str, object]]]
    resolved_records: dict[SegmentKey, list[dict[str, object]]]
    diagnostics: dict[SegmentKey, dict[str, object]]
    quality: dict[SegmentKey, QualityResult]
    attempt_rows: list[dict[str, object]]


def segment_key(record: dict[str, object]) -> SegmentKey:
    return str(record["video"]), int(record["segment_index"])


def frame_identity(record: dict[str, object]) -> tuple[str, int, int, str]:
    return (
        str(record["video"]),
        int(record["segment_index"]),
        int(record["frame_index"]),
        str(record["frame_path"]),
    )


def records_by_segment(
    records: Iterable[dict[str, object]],
) -> dict[SegmentKey, list[dict[str, object]]]:
    grouped: defaultdict[SegmentKey, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[segment_key(record)].append(record)
    return dict(grouped)


def filter_records_by_keys(
    records: Iterable[dict[str, object]],
    keys: set[SegmentKey],
) -> list[dict[str, object]]:
    return [record for record in records if segment_key(record) in keys]


def reusable_attempt_data(
    target_records: list[dict[str, object]],
    previous_records: list[dict[str, object]],
    previous_detections: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return new OCR inputs and detections reusable from a prior tier."""

    target_ids = {frame_identity(record) for record in target_records}
    previous_ids = {
        frame_identity(record) for record in previous_records
    }
    new_records = [
        record
        for record in target_records
        if frame_identity(record) not in previous_ids
    ]
    reusable_detections = [
        detection
        for detection in previous_detections
        if frame_identity(detection) in target_ids
    ]
    return new_records, reusable_detections


def _records_for_keys(
    rows: Iterable[dict[str, object]],
    keys: list[SegmentKey],
) -> dict[SegmentKey, list[dict[str, object]]]:
    grouped = records_by_segment(rows)
    return {key: grouped.get(key, []) for key in keys}


def _diagnostics_for_keys(
    diagnostics: Iterable[dict[str, object]],
    keys: list[SegmentKey],
) -> dict[SegmentKey, dict[str, object]]:
    grouped = {
        segment_key(diagnostic): diagnostic
        for diagnostic in diagnostics
    }
    return {key: grouped.get(key, {}) for key in keys}


def perform_attempt(
    *,
    keys: list[SegmentKey],
    attempt: int,
    tier: str,
    target_records: list[dict[str, object]],
    previous_records: dict[SegmentKey, list[dict[str, object]]],
    previous_detections: dict[SegmentKey, list[dict[str, object]]],
    ocr: object,
    config: PipelineConfig,
) -> AttemptOutcome:
    target_by_key = _records_for_keys(target_records, keys)
    new_records: list[dict[str, object]] = []
    reused_by_key: dict[SegmentKey, list[dict[str, object]]] = {}
    new_count_by_key: dict[SegmentKey, int] = {}

    for key in keys:
        new_for_key, reused_for_key = reusable_attempt_data(
            target_by_key[key],
            previous_records.get(key, []),
            previous_detections.get(key, []),
        )
        new_records.extend(new_for_key)
        reused_by_key[key] = reused_for_key
        new_count_by_key[key] = len(new_for_key)

    print(
        f"{tier}: OCR {len(new_records)} new frame(s) for "
        f"{len(keys)} segment(s)"
    )
    new_detections = (
        run_ocr(
            new_records,
            cache_dir=config.cache_dir,
            output_csv=None,
            min_score=config.min_score,
            batch_size=config.ocr_batch_size,
            ocr=ocr,
            progress_label=f"{tier} OCR",
        )
        if new_records
        else []
    )
    new_detections_by_key = records_by_segment(new_detections)
    attempt_detections = {
        key: (
            reused_by_key[key]
            + new_detections_by_key.get(key, [])
        )
        for key in keys
    }
    all_detections = [
        detection
        for key in keys
        for detection in attempt_detections[key]
    ]

    try:
        if all_detections:
            resolved, diagnostics = resolve_all_lineups(
                pd.DataFrame(all_detections),
                expected_players=config.players_per_lineup,
                min_number_count=config.min_number_count,
                max_gap_seconds=config.same_lineup_gap_seconds,
                signature_threshold=config.signature_threshold,
                enable_local_ocr=not config.disable_local_ocr,
            )
        else:
            resolved, diagnostics = [], []
    except LineupResolutionError as exc:
        resolved = []
        diagnostics = [
            {
                "video": key[0],
                "segment_index": key[1],
                "status": "unresolved",
                "resolution_method": "",
                "resolved_players": 0,
                "message": f"resolver error: {exc}",
            }
            for key in keys
        ]

    resolved_by_key = _records_for_keys(resolved, keys)
    diagnostics_by_key = _diagnostics_for_keys(diagnostics, keys)
    quality: dict[SegmentKey, QualityResult] = {}
    attempt_rows: list[dict[str, object]] = []
    for key in keys:
        diagnostic = diagnostics_by_key[key] or None
        result = evaluate_segment_quality(
            resolved_by_key[key],
            diagnostic,
            expected_players=config.players_per_lineup,
            min_pair_confidence=config.min_pair_confidence,
        )
        quality[key] = result
        resolver_status = (
            str(diagnostic.get("status", "unresolved"))
            if diagnostic is not None
            else "unresolved"
        )
        attempt_rows.append(
            {
                "video": key[0],
                "segment_index": key[1],
                "attempt": attempt,
                "tier": tier,
                "frame_count": len(target_by_key[key]),
                "new_ocr_frame_count": new_count_by_key[key],
                "detection_count": len(attempt_detections[key]),
                "resolver_status": resolver_status,
                "resolved_players": result.resolved_players,
                "quality_pass": result.passed,
                "quality_message": result.message,
            }
        )
        state = "PASS" if result.passed else "FAIL"
        print(
            f"  {key[0]}, segment {key[1]}: {state} - "
            f"{result.message}"
        )

    return AttemptOutcome(
        tier=tier,
        frame_records=target_by_key,
        detections=attempt_detections,
        resolved_records=resolved_by_key,
        diagnostics=diagnostics_by_key,
        quality=quality,
        attempt_rows=attempt_rows,
    )


def full_segment_records(
    records: list[dict[str, object]],
) -> list[dict[str, object]]:
    prepared: list[dict[str, object]] = []
    for record in records:
        prepared_record = dict(record)
        prepared_record.update(
            {
                "source_frame_path": record["frame_path"],
                "selection_layout": "fallback_full_segment",
                "selection_score": 0.0,
                "scout_frame_index": 0,
                "crop_side": "full",
                "crop_x1_norm": 0.0,
                "crop_x2_norm": 1.0,
            }
        )
        prepared.append(prepared_record)
    return prepared


def full_segment_selection(
    records: list[dict[str, object]],
    message: str,
) -> SegmentSelection:
    first = min(records, key=lambda row: float(row["timestamp_seconds"]))
    return SegmentSelection(
        video=str(first["video"]),
        segment_index=int(first["segment_index"]),
        layout="fallback_full_segment",
        status="fallback",
        score=0.0,
        scout_frame_index=0,
        scout_timestamp_seconds=float(first["timestamp_seconds"]),
        formation_anchor_count=0,
        table_pair_count=0,
        number_count=0,
        name_count=0,
        crop_x1_norm=0.0,
        crop_x2_norm=1.0,
        selected_frame_indices=tuple(
            sorted(int(record["frame_index"]) for record in records)
        ),
        message=message,
    )


def selection_maps(
    selections: list[SegmentSelection],
) -> tuple[dict[SegmentKey, SegmentSelection], set[SegmentKey]]:
    selected: dict[SegmentKey, SegmentSelection] = {}
    fallback: set[SegmentKey] = set()
    for selection in selections:
        key = (selection.video, selection.segment_index)
        if selection.status == "selected":
            selected[key] = selection
        else:
            fallback.add(key)
    return selected, fallback


def final_diagnostic(
    key: SegmentKey,
    diagnostic: dict[str, object],
    quality: QualityResult,
    tier: str,
) -> dict[str, object]:
    original_message = str(diagnostic.get("message", "")).strip()
    quality_message = (
        f"quality gate passed at {tier}: {quality.message}"
        if quality.passed
        else f"quality gate failed after {tier}: {quality.message}"
    )
    message = "; ".join(
        part for part in (original_message, quality_message) if part
    )
    return {
        "video": key[0],
        "segment_index": key[1],
        "status": "resolved" if quality.passed else "unresolved",
        "resolution_method": (
            str(diagnostic.get("resolution_method", ""))
            if quality.passed
            else ""
        ),
        "resolved_players": quality.resolved_players,
        "message": message,
    }
