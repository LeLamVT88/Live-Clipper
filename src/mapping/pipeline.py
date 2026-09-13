from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from .config import PipelineConfig
from .engine import create_ocr, run_ocr, write_csv
from .frames import (LineupOCRError, SegmentKey, extract_segment_frames,
                     group_records_by_segment, load_lineup_clips, segment_key)
from .resolver import (LineupResolutionError, QualityResult,
                       evaluate_segment_quality, resolve_all_lineups)
from . import schema
from .selection_io import materialize_selected_frames, selection_diagnostic_rows
from .selector import (
    SegmentSelection,
    fallback_selection,
    sample_scout_frames,
    select_segment_frames,
)


Record = dict[str, object]
RecordList = list[Record]
GroupedRecords = dict[SegmentKey, RecordList]
DiagnosticMap = dict[SegmentKey, Record]


@dataclass(slots=True)
class AttemptOutcome:
    tier: str
    frame_records: GroupedRecords
    detections: GroupedRecords
    resolved_records: GroupedRecords
    diagnostics: DiagnosticMap
    quality: dict[SegmentKey, QualityResult]
    attempt_rows: RecordList


@dataclass(slots=True)
class WorkflowState:
    ordered_keys: list[SegmentKey]
    active_frames: GroupedRecords = field(default_factory=dict)
    active_detections: GroupedRecords = field(default_factory=dict)
    final_records: GroupedRecords = field(default_factory=dict)
    final_diagnostics: DiagnosticMap = field(default_factory=dict)
    final_quality: dict[SegmentKey, QualityResult] = field(default_factory=dict)
    final_tier: dict[SegmentKey, str] = field(default_factory=dict)
    attempt_rows: RecordList = field(default_factory=list)
    final_selection: dict[SegmentKey, SegmentSelection] = field(default_factory=dict)
    def apply(
        self, outcome: AttemptOutcome, selections: dict[SegmentKey, SegmentSelection],
        *, final: bool = False,
    ) -> set[SegmentKey]:
        self.active_frames.update(outcome.frame_records)
        self.active_detections.update(outcome.detections)
        self.final_selection.update(selections)
        self.attempt_rows.extend(outcome.attempt_rows)
        failed = {key for key, quality in outcome.quality.items() if not quality.passed}
        for key, quality in outcome.quality.items():
            if quality.passed:
                self.final_records[key] = outcome.resolved_records[key]
            if quality.passed or final:
                self.final_diagnostics[key] = outcome.diagnostics[key]
                self.final_quality[key] = quality
                self.final_tier[key] = outcome.tier
        return failed


def frame_identity(record: Record) -> tuple[str, int, int, str]:
    return (
        str(record["video"]), int(record["segment_index"]),
        int(record["frame_index"]), str(record["frame_path"]),
    )


def reusable_attempt_data(
    target_records: RecordList, previous_records: RecordList,
    previous_detections: RecordList,
) -> tuple[RecordList, RecordList]:
    target_ids = {frame_identity(record) for record in target_records}
    previous_ids = {frame_identity(record) for record in previous_records}
    return (
        [record for record in target_records if frame_identity(record) not in previous_ids],
        [row for row in previous_detections if frame_identity(row) in target_ids],
    )


def _for_keys(rows: Iterable[Record], keys: list[SegmentKey]) -> GroupedRecords:
    grouped = group_records_by_segment(rows)
    return {key: grouped.get(key, []) for key in keys}


def _resolve(
    keys: list[SegmentKey], detections: GroupedRecords, config: PipelineConfig
) -> tuple[GroupedRecords, DiagnosticMap]:
    flat = [row for key in keys for row in detections[key]]
    try:
        if not flat:
            return _for_keys([], keys), {key: {} for key in keys}
        resolved, diagnostics = resolve_all_lineups(
            pd.DataFrame(flat), config.players_per_lineup, config.min_number_count,
            config.same_lineup_gap_seconds, config.signature_threshold,
            enable_local_ocr=not config.disable_local_ocr,
        )
    except LineupResolutionError as exc:
        resolved = []
        diagnostics = [
            {
                "video": video,
                "segment_index": index,
                "status": "unresolved",
                "resolution_method": "",
                "resolved_players": 0,
                "message": f"resolver error: {exc}",
            }
            for video, index in keys
        ]
    diagnostic_map = {segment_key(row): row for row in diagnostics}
    return _for_keys(resolved, keys), {key: diagnostic_map.get(key, {}) for key in keys}


def perform_attempt(
    *, keys: list[SegmentKey], attempt: int, tier: str,
    target_records: RecordList, previous_records: GroupedRecords,
    previous_detections: GroupedRecords, ocr: object, config: PipelineConfig,
) -> AttemptOutcome:
    targets = _for_keys(target_records, keys)
    new_records: RecordList = []
    reused: GroupedRecords = {}
    new_counts: dict[SegmentKey, int] = {}
    for key in keys:
        new, old = reusable_attempt_data(
            targets[key], previous_records.get(key, []), previous_detections.get(key, [])
        )
        new_records.extend(new)
        reused[key], new_counts[key] = old, len(new)
    print(f"{tier}: OCR {len(new_records)} new frame(s) for {len(keys)} segment(s)")
    new_detections = (
        run_ocr(
            new_records, config.cache_dir, None, config.min_score,
            config.ocr_batch_size, ocr, f"{tier} OCR",
        )
        if new_records
        else []
    )
    fresh = group_records_by_segment(new_detections)
    detections = {key: reused[key] + fresh.get(key, []) for key in keys}
    resolved, diagnostics = _resolve(keys, detections, config)
    quality: dict[SegmentKey, QualityResult] = {}
    attempt_rows: RecordList = []
    for key in keys:
        diagnostic = diagnostics[key] or None
        result = evaluate_segment_quality(
            resolved[key], diagnostic, config.players_per_lineup, config.min_pair_confidence
        )
        quality[key] = result
        attempt_rows.append(
            {
                "video": key[0], "segment_index": key[1],
                "attempt": attempt, "tier": tier,
                "frame_count": len(targets[key]), "new_ocr_frame_count": new_counts[key],
                "detection_count": len(detections[key]),
                "resolver_status": str((diagnostic or {}).get("status", "unresolved")),
                "resolved_players": result.resolved_players, "quality_pass": result.passed,
                "quality_message": result.message,
            }
        )
        status = "PASS" if result.passed else "FAIL"
        print(f"  {key[0]}, segment {key[1]}: {status} - {result.message}")
    return AttemptOutcome(tier, targets, detections, resolved, diagnostics, quality, attempt_rows)


def _full_records(records: RecordList) -> RecordList:
    metadata = {
        "selection_layout": "fallback_full_segment", "selection_score": 0.0,
        "scout_frame_index": 0, "crop_side": "full",
        "crop_x1_norm": 0.0, "crop_x2_norm": 1.0,
    }
    return [dict(row, source_frame_path=row["frame_path"], **metadata) for row in records]


class LineupWorkflow:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.frames: RecordList = []
        self.scout = pd.DataFrame()
        self.ocr: object | None = None
        self.state: WorkflowState | None = None
    def run(self) -> int:
        self.frames = self._extract_frames()
        if self.config.extract_only:
            return 0
        if not self.frames:
            raise LineupOCRError("No lineup frames were extracted.")
        self.state = WorkflowState(list(group_records_by_segment(self.frames)))
        self.scout, self.ocr = self._run_scout()
        pending = set(self._state().ordered_keys)
        full: set[SegmentKey] = set()
        tiers = self.config.initial_frame_count, self.config.expanded_frame_count
        for attempt, count in enumerate(tiers, start=1):
            if not pending:
                break
            pending, direct_full = self._run_selected_tier(pending, attempt, count)
            full.update(direct_full)
        full.update(pending)
        if full:
            self._run_full_tier(full)
        self._write_outputs()
        state = self._state()
        passed = sum(result.passed for result in state.final_quality.values())
        players = sum(map(len, state.final_records.values()))
        print(f"Pipeline complete: {passed}/{len(state.ordered_keys)} segment(s) passed "
              f"the quality gate; {players} player rows.")
        print(f"Resolved lineups: {self.config.resolved_output_csv}")
        print(f"Attempt diagnostics: {self.config.attempts_csv}")
        return 0 if passed == len(state.ordered_keys) else 2
    def _extract_frames(self) -> RecordList:
        segments = load_lineup_clips(self.config.clips_dir)
        records: RecordList = []
        for position, segment in enumerate(segments, start=1):
            print(
                f"[{position}/{len(segments)}] Extracting lineup clip "
                f"{segment.video_path.name}: {segment.start_seconds:.3f}-"
                f"{segment.end_seconds:.3f}s at {self.config.fps:g} FPS"
            )
            records.extend(
                extract_segment_frames(
                    segment, self.config.frames_dir, self.config.fps, self.config.jpeg_quality
                )
            )
        write_csv(records, self.config.frames_csv, schema.FRAME_COLUMNS)
        print(f"Extracted {len(records)} frame(s): {self.config.frames_csv}")
        return records
    def _run_scout(self) -> tuple[pd.DataFrame, object]:
        frames = sample_scout_frames(self.frames, self.config.scout_fps)
        print(
            f"Scout selection: {len(frames)}/{len(self.frames)} full frame(s) "
            f"at {self.config.scout_fps:g} FPS"
        )
        print(f"Loading OCR models from: {self.config.cache_dir}")
        ocr = create_ocr(self.config.cache_dir, self.config.ocr_batch_size)
        detections = run_ocr(
            frames, self.config.cache_dir, self.config.scout_output_csv,
            self.config.min_score, self.config.ocr_batch_size, ocr, "Scout OCR",
        )
        return pd.DataFrame(detections), ocr
    def _run_selected_tier(
        self, keys: set[SegmentKey], attempt: int, frame_count: int
    ) -> tuple[set[SegmentKey], set[SegmentKey]]:
        source = [row for row in self.frames if segment_key(row) in keys]
        selections = select_segment_frames(source, self.scout, frame_count, self.config.scout_fps)
        selected = {(item.video, item.segment_index): item for item in selections
                    if item.status == "selected"}
        direct_full = {
            (item.video, item.segment_index) for item in selections if item.status != "selected"
        }
        ordered = [key for key in self._state().ordered_keys if key in selected]
        if not ordered:
            return set(), direct_full
        frames = materialize_selected_frames(
            source, [selected[key] for key in ordered],
            self.config.selected_frames_dir, self.config.jpeg_quality,
        )
        outcome = self._attempt(ordered, attempt, f"tier{attempt}_selected_{frame_count}", frames)
        return self._state().apply(outcome, selected), direct_full
    def _run_full_tier(self, keys: set[SegmentKey]) -> None:
        source = [row for row in self.frames if segment_key(row) in keys]
        grouped = group_records_by_segment(source)
        message = "Using the complete 2 FPS segment after selection or quality fallback."
        selections = {key: fallback_selection(key, rows, message)
                      for key, rows in grouped.items()}
        ordered = [key for key in self._state().ordered_keys if key in keys]
        outcome = self._attempt(ordered, 3, "tier3_full_2fps", _full_records(source))
        self._state().apply(outcome, selections, final=True)
    def _attempt(
        self, keys: list[SegmentKey], attempt: int, tier: str, frames: RecordList
    ) -> AttemptOutcome:
        state = self._state()
        if self.ocr is None:
            raise LineupOCRError("OCR model is not initialized.")
        return perform_attempt(
            keys=keys, attempt=attempt, tier=tier, target_records=frames,
            previous_records=state.active_frames,
            previous_detections=state.active_detections,
            ocr=self.ocr, config=self.config,
        )
    def _write_outputs(self) -> None:
        state = self._state()
        missing = set(state.ordered_keys) - set(state.final_quality)
        if missing:
            labels = ", ".join(f"{video} segment {index}" for video, index in sorted(missing))
            raise LineupOCRError(f"Pipeline did not produce a final quality result for: {labels}")
        def flatten(grouped: GroupedRecords) -> RecordList:
            return [row for key in state.ordered_keys for row in grouped.get(key, [])]
        diagnostics = [self._final_diagnostic(key) for key in state.ordered_keys]
        outputs = (
            (flatten(state.active_detections), self.config.output_csv, schema.DETECTION_COLUMNS),
            (flatten(state.active_frames), self.config.selected_frames_csv, schema.SELECTED_FRAME_COLUMNS),
            (
                selection_diagnostic_rows(
                    [state.final_selection[key] for key in state.ordered_keys]
                ),
                self.config.selection_diagnostics_csv,
                schema.SELECTION_DIAGNOSTIC_COLUMNS,
            ),
            (flatten(state.final_records), self.config.resolved_output_csv, schema.RESOLVED_COLUMNS),
            (diagnostics, self.config.resolved_diagnostics_csv, schema.DIAGNOSTIC_COLUMNS),
            (state.attempt_rows, self.config.attempts_csv, schema.ATTEMPT_COLUMNS),
        )
        for rows, path, columns in outputs:
            write_csv(rows, path, columns)
    def _final_diagnostic(self, key: SegmentKey) -> Record:
        state = self._state()
        diagnostic = state.final_diagnostics.get(key, {})
        quality, tier = state.final_quality[key], state.final_tier[key]
        quality_message = (
            f"quality gate passed at {tier}: {quality.message}"
            if quality.passed
            else f"quality gate failed after {tier}: {quality.message}"
        )
        message = "; ".join(
            part for part in (str(diagnostic.get("message", "")).strip(), quality_message) if part
        )
        return {
            "video": key[0], "segment_index": key[1],
            "status": "resolved" if quality.passed else "unresolved",
            "resolution_method": (
                str(diagnostic.get("resolution_method", "")) if quality.passed else ""
            ),
            "resolved_players": quality.resolved_players, "message": message,
        }
    def _state(self) -> WorkflowState:
        if self.state is None:
            raise LineupOCRError("Workflow state is not initialized.")
        return self.state
