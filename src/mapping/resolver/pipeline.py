from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .common import LineupResolutionError
from .formation import attempt_formation_resolution
from .layout import formation_refinement_frames
from .local_models import create_local_number_recognizer, create_local_table_ocr
from .player_names import enrich_player_names
from .refinement import refine_formation_numbers, refine_table_numbers
from .table import resolve_table_events


@dataclass(slots=True)
class ResolutionResult:
    records: list[dict[str, object]]
    method: str = ""


@dataclass(slots=True)
class LocalOCRModels:
    table_ocr: object | None = None
    number_recognizer: object | None = None
    def get_table_ocr(self, purpose: str) -> object:
        if self.table_ocr is None:
            print(f"Loading PaddleOCR for local {purpose} refinement...")
            self.table_ocr = create_local_table_ocr()
        return self.table_ocr
    def get_number_recognizer(self) -> object:
        if self.number_recognizer is None:
            self.number_recognizer = create_local_number_recognizer()
        return self.number_recognizer


LOCAL_OCR_ERRORS = (ImportError, LineupResolutionError, ModuleNotFoundError, OSError, ValueError)


def resolve_table(
    segment: pd.DataFrame, *, expected_players: int, enable_local_ocr: bool,
    expected_lineups: int, max_gap_seconds: float, signature_threshold: float,
    models: LocalOCRModels, messages: list[str],
) -> tuple[ResolutionResult, pd.DataFrame]:
    table_records, table_count = resolve_table_events(
        segment, expected_players, max_gap_seconds, signature_threshold
    )
    expected_total = expected_players * expected_lineups
    if len(table_records) == expected_total:
        messages.append("Complete repeated table/list consensus.")
        return ResolutionResult(table_records, "table"), segment
    if table_records or table_count:
        messages.append(
            f"table/list pass resolved {len(table_records) // expected_players}/"
            f"{expected_lineups} lineup(s); best event {table_count}/{expected_players} players"
        )
    can_refine = enable_local_ocr and len(table_records) < expected_total
    can_refine &= table_count >= 4
    can_refine &= "frame_path" in segment.columns
    if not can_refine:
        return ResolutionResult([]), segment
    try:
        refined, added_count = refine_table_numbers(
            segment, models.get_table_ocr("table-number"), expected_players
        )
        if not added_count:
            return ResolutionResult([]), segment
        messages.append(f"local table OCR added {added_count} number observations")
        table_records, _ = resolve_table_events(
            refined, expected_players, max_gap_seconds, signature_threshold
        )
        if len(table_records) == expected_total:
            for record in table_records:
                record["resolution_method"] = "table+local_ocr"
            return ResolutionResult(table_records, "table+local_ocr"), refined
        return ResolutionResult([]), refined
    except LOCAL_OCR_ERRORS as exc:
        messages.append(f"local table OCR unavailable: {exc}")
        return ResolutionResult([]), segment


def refine_formation(
    segment: pd.DataFrame, *, enable_local_ocr: bool,
    models: LocalOCRModels, messages: list[str],
) -> pd.DataFrame:
    can_refine = enable_local_ocr and "frame_path" in segment.columns
    can_refine &= bool(formation_refinement_frames(segment))
    if not can_refine:
        return segment
    try:
        refined, added_count = refine_formation_numbers(
            segment,
            ocr=models.get_table_ocr("formation-number"),
            recognizer=models.get_number_recognizer(),
        )
        if added_count:
            messages.append(f"local formation OCR added {added_count} number observations")
            return refined
    except LOCAL_OCR_ERRORS as exc:
        messages.append(f"local formation OCR unavailable: {exc}")
    return segment


def resolve_formation(
    segment: pd.DataFrame, *, expected_players: int, min_number_count: int,
    max_gap_seconds: float, signature_threshold: float, enable_local_ocr: bool,
    models: LocalOCRModels, messages: list[str], expected_lineups: int = 1,
) -> ResolutionResult:
    resolution_args = {
        "expected_players": expected_players,
        "min_number_count": min_number_count,
        "max_gap_seconds": max_gap_seconds,
        "signature_threshold": signature_threshold,
    }
    _, initial_events, initial_records, initial_errors = attempt_formation_resolution(
        segment, **resolution_args
    )
    if (
        initial_events
        and not initial_errors
        and len(initial_records) == expected_players * expected_lineups
    ):
        return ResolutionResult(initial_records, "formation")
    refined = refine_formation(
        segment, enable_local_ocr=enable_local_ocr, models=models, messages=messages
    )
    if refined is segment:
        events = initial_events
        records = initial_records
        event_errors = initial_errors
    else:
        _, events, records, event_errors = attempt_formation_resolution(refined, **resolution_args)
    if not events:
        messages.append("no formation snapshot found")
        return ResolutionResult([])
    refined_locally = any(msg.startswith("local formation OCR added") for msg in messages)
    method = "formation+local_ocr" if refined_locally else "formation"
    for record in records:
        record["resolution_method"] = method
    messages.extend(
        (f"ignored incomplete formation candidate: {error}" if records else error)
        for error in event_errors
    )
    return ResolutionResult(records, method if records else "")


def resolve_all_lineups(
    detections: pd.DataFrame,
    expected_players: int,
    min_number_count: int,
    max_gap_seconds: float,
    signature_threshold: float,
    enable_local_ocr: bool = True,
    reference_detections: pd.DataFrame | None = None,
    expected_lineups: int = 1,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    models = LocalOCRModels()
    grouped = detections.groupby(["video", "segment_index"], sort=False)
    for (video, segment_index), segment in grouped:
        messages: list[str] = []
        result, segment = resolve_table(
            segment, expected_players=expected_players, enable_local_ocr=enable_local_ocr,
            expected_lineups=expected_lineups, max_gap_seconds=max_gap_seconds,
            signature_threshold=signature_threshold,
            models=models, messages=messages,
        )
        if not result.records:
            result = resolve_formation(
                segment, expected_players=expected_players, min_number_count=min_number_count,
                max_gap_seconds=max_gap_seconds, signature_threshold=signature_threshold,
                enable_local_ocr=enable_local_ocr, models=models, messages=messages,
                expected_lineups=expected_lineups,
            )
        for record in result.records:
            record.setdefault(
                "number_source",
                "table" if result.method.startswith("table") else "detector",
            )
            record.setdefault("name_source", "resolver")
        if (
            result.records
            and reference_detections is not None
            and not reference_detections.empty
        ):
            reference_segment = reference_detections[
                (reference_detections["video"] == video)
                & (reference_detections["segment_index"] == segment_index)
            ]
            result.records = enrich_player_names(result.records, reference_segment)
        records.extend(result.records)
        diagnostics.append({
            "video": video, "segment_index": int(segment_index),
            "status": "resolved" if result.records else "unresolved",
            "resolution_method": result.method, "resolved_players": len(result.records),
            "message": "; ".join(messages),
        })
        label = f"{video}, segment {int(segment_index)}"
        summary = (f"{len(result.records)} players resolved from {result.method}"
                   if result.records else "unresolved")
        print(f"{label}: {summary}")
    return records, diagnostics
