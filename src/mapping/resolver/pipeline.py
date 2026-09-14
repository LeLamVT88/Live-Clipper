from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .common import (
    LineupResolutionError,
    detections_before_substitutes_by_scene,
    detections_without_substitute_panel_by_scene,
)
from .formation import attempt_formation_resolution
from .layout import formation_recovery_windows, formation_refinement_frames
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
    table_segment = (
        detections_before_substitutes_by_scene(segment)
        if expected_lineups > 1
        else segment
    )
    table_records, table_count = resolve_table_events(
        table_segment, expected_players, max_gap_seconds, signature_threshold
    )
    expected_total = expected_players * expected_lineups
    if len(table_records) == expected_total:
        messages.append("Complete repeated table/list consensus.")
        return ResolutionResult(table_records, "table"), table_segment
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
            table_segment, models.get_table_ocr("table-number"), expected_players
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
        return ResolutionResult([]), segment
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
    enable_failed_recovery: bool = False,
) -> ResolutionResult:
    formation_segment = (
        detections_without_substitute_panel_by_scene(segment)
        if expected_lineups > 1
        else segment
    )
    resolution_args = {
        "expected_players": expected_players,
        "min_number_count": min_number_count,
        "max_gap_seconds": max_gap_seconds,
        "signature_threshold": signature_threshold,
    }
    _, initial_events, initial_records, initial_errors = attempt_formation_resolution(
        formation_segment, **resolution_args
    )
    if (
        initial_events
        and not initial_errors
        and len(initial_records) == expected_players * expected_lineups
    ):
        return ResolutionResult(initial_records, "formation")
    refined = refine_formation(
        formation_segment,
        enable_local_ocr=enable_local_ocr,
        models=models,
        messages=messages,
    )
    if refined is formation_segment:
        events = initial_events
        records = initial_records
        event_errors = initial_errors
    else:
        _, events, records, event_errors = attempt_formation_resolution(refined, **resolution_args)
    if not events:
        messages.append("no formation snapshot found")
    refined_locally = any(msg.startswith("local formation OCR added") for msg in messages)
    method = "formation+local_ocr" if refined_locally else "formation"
    for record in records:
        record["resolution_method"] = method
    messages.extend(
        (f"ignored incomplete formation candidate: {error}" if records else error)
        for error in event_errors
    )
    if records or not enable_failed_recovery or not enable_local_ocr:
        return ResolutionResult(records, method if records else "")
    if expected_lineups != 1 or "frame_path" not in segment.columns:
        return ResolutionResult([])
    windows = formation_recovery_windows(segment, expected_players)
    if not windows:
        messages.append("stable name-based formation recovery found no eligible window")
        return ResolutionResult([])
    recovery_errors: list[str] = []
    for position, window in enumerate(windows, start=1):
        frame_indices = [frame_index for frame_index, _ in window]
        window_segment = segment[segment["frame_index"].isin(frame_indices)].copy()
        try:
            recovered, added_count = refine_formation_numbers(
                window_segment,
                ocr=models.get_table_ocr("stable-formation-number"),
                recognizer=models.get_number_recognizer(),
                chosen_frames=window,
                recover_all_names=True,
            )
            if not added_count:
                recovery_errors.append(f"window {position} added no number observations")
                continue
            _, _, recovered_records, errors = attempt_formation_resolution(
                recovered, **resolution_args
            )
            if len(recovered_records) != expected_players:
                detail = errors[-1] if errors else (
                    f"resolved {len(recovered_records)}/{expected_players} players"
                )
                recovery_errors.append(f"window {position}: {detail}")
                continue
            for record in recovered_records:
                record["resolution_method"] = "formation+stable_name_recovery"
            timestamps = [
                float(window_segment[window_segment["frame_index"] == frame_index][
                    "timestamp_seconds"
                ].iloc[0])
                for frame_index in frame_indices
            ]
            messages.append(
                f"stable name-based formation recovery resolved {expected_players} "
                "players from "
                f"window {min(timestamps):.1f}-{max(timestamps):.1f}s; "
                f"added {added_count} number observations"
            )
            return ResolutionResult(
                recovered_records, "formation+stable_name_recovery"
            )
        except LOCAL_OCR_ERRORS as exc:
            recovery_errors.append(f"window {position}: {exc}")
    messages.append(
        "stable name-based formation recovery failed: " + "; ".join(recovery_errors)
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
    enable_failed_formation_recovery: bool = False,
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
                enable_failed_recovery=enable_failed_formation_recovery,
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
