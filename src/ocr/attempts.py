"""Run and evaluate one adaptive OCR attempt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .config import PipelineConfig
from .frames import (
    SegmentKey,
    group_records_by_segment,
    segment_key,
)
from .ocr_engine import run_ocr
from .resolver import (
    LineupResolutionError,
    QualityResult,
    evaluate_segment_quality,
    resolve_all_lineups,
)
from .schema import ATTEMPT_COLUMNS
from .selector import SegmentSelection


Record = dict[str, object]
RecordList = list[Record]
GroupedRecords = dict[SegmentKey, RecordList]
DiagnosticMap = dict[SegmentKey, Record]

@dataclass
class AttemptOutcome:
    tier: str
    frame_records: GroupedRecords
    detections: GroupedRecords
    resolved_records: GroupedRecords
    diagnostics: DiagnosticMap
    quality: dict[SegmentKey, QualityResult]
    attempt_rows: RecordList


def frame_identity(record: Record) -> tuple[str, int, int, str]:
    return (
        str(record["video"]),
        int(record["segment_index"]),
        int(record["frame_index"]),
        str(record["frame_path"]),
    )


def filter_records_by_keys(
    records: Iterable[Record],
    keys: set[SegmentKey],
) -> RecordList:
    return [record for record in records if segment_key(record) in keys]


def reusable_attempt_data(
    target_records: RecordList,
    previous_records: RecordList,
    previous_detections: RecordList,
) -> tuple[RecordList, RecordList]:
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
    rows: Iterable[Record],
    keys: list[SegmentKey],
) -> GroupedRecords:
    grouped = group_records_by_segment(rows)
    return {key: grouped.get(key, []) for key in keys}


def _diagnostics_for_keys(
    diagnostics: Iterable[Record],
    keys: list[SegmentKey],
) -> DiagnosticMap:
    grouped = {
        segment_key(diagnostic): diagnostic
        for diagnostic in diagnostics
    }
    return {key: grouped.get(key, {}) for key in keys}


def _run_attempt_ocr(
    *,
    keys: list[SegmentKey],
    tier: str,
    target_records: RecordList,
    previous_records: GroupedRecords,
    previous_detections: GroupedRecords,
    ocr: object,
    config: PipelineConfig,
) -> tuple[GroupedRecords, GroupedRecords, dict[SegmentKey, int]]:
    target_by_key = _records_for_keys(target_records, keys)
    new_records: RecordList = []
    reused_by_key: GroupedRecords = {}
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
    new_detections_by_key = group_records_by_segment(new_detections)
    attempt_detections = {
        key: reused_by_key[key] + new_detections_by_key.get(key, [])
        for key in keys
    }
    return target_by_key, attempt_detections, new_count_by_key


def _resolve_attempt(
    keys: list[SegmentKey],
    detections_by_key: GroupedRecords,
    config: PipelineConfig,
) -> tuple[GroupedRecords, DiagnosticMap]:
    all_detections = [
        detection
        for key in keys
        for detection in detections_by_key[key]
    ]
    try:
        resolved, diagnostics = (
            resolve_all_lineups(
                pd.DataFrame(all_detections),
                expected_players=config.players_per_lineup,
                min_number_count=config.min_number_count,
                max_gap_seconds=config.same_lineup_gap_seconds,
                signature_threshold=config.signature_threshold,
                enable_local_ocr=not config.disable_local_ocr,
            )
            if all_detections
            else ([], [])
        )
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
    return (
        _records_for_keys(resolved, keys),
        _diagnostics_for_keys(diagnostics, keys),
    )


def _evaluate_attempt(
    *,
    keys: list[SegmentKey],
    attempt: int,
    tier: str,
    target_by_key: GroupedRecords,
    detections_by_key: GroupedRecords,
    new_count_by_key: dict[SegmentKey, int],
    resolved_by_key: GroupedRecords,
    diagnostics_by_key: DiagnosticMap,
    config: PipelineConfig,
) -> tuple[dict[SegmentKey, QualityResult], RecordList]:
    quality: dict[SegmentKey, QualityResult] = {}
    attempt_rows: RecordList = []
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
                "detection_count": len(detections_by_key[key]),
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
    return quality, attempt_rows


def perform_attempt(
    *,
    keys: list[SegmentKey],
    attempt: int,
    tier: str,
    target_records: RecordList,
    previous_records: GroupedRecords,
    previous_detections: GroupedRecords,
    ocr: object,
    config: PipelineConfig,
) -> AttemptOutcome:
    target_by_key, detections_by_key, new_count_by_key = (
        _run_attempt_ocr(
            keys=keys,
            tier=tier,
            target_records=target_records,
            previous_records=previous_records,
            previous_detections=previous_detections,
            ocr=ocr,
            config=config,
        )
    )
    resolved_by_key, diagnostics_by_key = _resolve_attempt(
        keys,
        detections_by_key,
        config,
    )
    quality, attempt_rows = _evaluate_attempt(
        keys=keys,
        attempt=attempt,
        tier=tier,
        target_by_key=target_by_key,
        detections_by_key=detections_by_key,
        new_count_by_key=new_count_by_key,
        resolved_by_key=resolved_by_key,
        diagnostics_by_key=diagnostics_by_key,
        config=config,
    )
    return AttemptOutcome(
        tier=tier,
        frame_records=target_by_key,
        detections=detections_by_key,
        resolved_records=resolved_by_key,
        diagnostics=diagnostics_by_key,
        quality=quality,
        attempt_rows=attempt_rows,
    )


def full_segment_records(
    records: RecordList,
) -> RecordList:
    prepared: RecordList = []
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
    records: RecordList,
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
) -> Record:
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
