"""Coordinate extraction, adaptive OCR, and lineup resolution."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .attempts import (
    AttemptOutcome,
    filter_records_by_keys,
    final_diagnostic,
    full_segment_records,
    full_segment_selection,
    perform_attempt,
    selection_maps,
)
from .config import PipelineConfig
from .frames import (
    LineupOCRError,
    SegmentKey,
    extract_segment_frames,
    group_records_by_segment,
    load_segments,
)
from .ocr_engine import (
    create_ocr,
    run_ocr,
    write_csv,
)
from .resolver import (
    QualityResult,
)
from .schema import (
    ATTEMPT_COLUMNS,
    DETECTION_COLUMNS,
    DIAGNOSTIC_COLUMNS,
    FRAME_COLUMNS,
    RESOLVED_COLUMNS,
    SELECTED_FRAME_COLUMNS,
    SELECTION_DIAGNOSTIC_COLUMNS,
)
from .selector import (
    SegmentSelection,
    sample_scout_frames,
    select_segment_frames,
)
from .selection_io import (
    materialize_selected_frames,
    selection_diagnostic_rows,
)


@dataclass
class WorkflowState:
    ordered_keys: list[SegmentKey]
    active_frames: dict[
        SegmentKey, list[dict[str, object]]
    ] = field(default_factory=dict)
    active_detections: dict[
        SegmentKey, list[dict[str, object]]
    ] = field(default_factory=dict)
    final_records: dict[
        SegmentKey, list[dict[str, object]]
    ] = field(default_factory=dict)
    final_diagnostics: dict[
        SegmentKey, dict[str, object]
    ] = field(default_factory=dict)
    final_quality: dict[SegmentKey, QualityResult] = field(
        default_factory=dict
    )
    final_tier: dict[SegmentKey, str] = field(default_factory=dict)
    final_selection: dict[SegmentKey, SegmentSelection] = field(
        default_factory=dict
    )
    attempt_rows: list[dict[str, object]] = field(default_factory=list)

    def apply(
        self,
        outcome: AttemptOutcome,
        selections: dict[SegmentKey, SegmentSelection],
    ) -> set[SegmentKey]:
        self.active_frames.update(outcome.frame_records)
        self.active_detections.update(outcome.detections)
        self.attempt_rows.extend(outcome.attempt_rows)
        # Keep the most recent selection even when the quality gate fails.
        # Final diagnostics must still describe unresolved tier-3 segments.
        self.final_selection.update(selections)

        failed: set[SegmentKey] = set()
        for key, quality in outcome.quality.items():
            if not quality.passed:
                failed.add(key)
                continue
            self.final_records[key] = outcome.resolved_records[key]
            self.final_diagnostics[key] = outcome.diagnostics[key]
            self.final_quality[key] = quality
            self.final_tier[key] = outcome.tier
        return failed

    def keep_final_failures(self, outcome: AttemptOutcome) -> None:
        for key, quality in outcome.quality.items():
            if quality.passed:
                continue
            self.final_diagnostics[key] = outcome.diagnostics[key]
            self.final_quality[key] = quality
            self.final_tier[key] = outcome.tier


class LineupWorkflow:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.frames: list[dict[str, object]] = []
        self.scout = pd.DataFrame()
        self.ocr: object | None = None
        self.state: WorkflowState | None = None

    def run(self) -> int:
        self.frames = self._extract_frames()
        if self.config.extract_only:
            return 0
        if not self.frames:
            raise LineupOCRError("No lineup frames were extracted.")

        self.state = WorkflowState(
            ordered_keys=list(group_records_by_segment(self.frames))
        )
        self.scout, self.ocr = self._run_scout()

        all_keys = set(self.state.ordered_keys)
        tier2_keys, tier3_keys = self._run_selected_tier(
            all_keys,
            attempt=1,
            frame_count=self.config.initial_frame_count,
        )
        if tier2_keys:
            tier2_failed, tier2_direct_full = self._run_selected_tier(
                tier2_keys,
                attempt=2,
                frame_count=self.config.expanded_frame_count,
            )
            tier3_keys.update(tier2_failed)
            tier3_keys.update(tier2_direct_full)
        if tier3_keys:
            self._run_full_tier(tier3_keys)

        self._validate_completion()
        self._write_outputs()
        return self._print_summary()

    def _extract_frames(self) -> list[dict[str, object]]:
        segments = load_segments(
            self.config.segments_csv,
            self.config.video_dir,
        )
        records: list[dict[str, object]] = []
        for position, segment in enumerate(segments, start=1):
            print(
                f"[{position}/{len(segments)}] Extracting "
                f"{segment.video_path.name} segment {segment.index}: "
                f"{segment.start_seconds:.3f}-{segment.end_seconds:.3f}s "
                f"at {self.config.fps:g} FPS"
            )
            records.extend(
                extract_segment_frames(
                    segment,
                    frames_dir=self.config.frames_dir,
                    fps=self.config.fps,
                    jpeg_quality=self.config.jpeg_quality,
                )
            )
        write_csv(
            records,
            self.config.frames_csv,
            FRAME_COLUMNS,
        )
        print(
            f"Extracted {len(records)} frame(s): "
            f"{self.config.frames_csv}"
        )
        return records

    def _run_scout(self) -> tuple[pd.DataFrame, object]:
        scout_frames = sample_scout_frames(
            self.frames,
            scout_fps=self.config.scout_fps,
        )
        print(
            f"Scout selection: {len(scout_frames)}/{len(self.frames)} "
            f"full frame(s) at {self.config.scout_fps:g} FPS"
        )
        print(f"Loading OCR models from: {self.config.cache_dir}")
        ocr = create_ocr(
            self.config.cache_dir,
            recognition_batch_size=self.config.ocr_batch_size,
        )
        detections = run_ocr(
            scout_frames,
            cache_dir=self.config.cache_dir,
            output_csv=self.config.scout_output_csv,
            min_score=self.config.min_score,
            batch_size=self.config.ocr_batch_size,
            ocr=ocr,
            progress_label="Scout OCR",
        )
        return pd.DataFrame(detections), ocr

    def _run_selected_tier(
        self,
        target_keys: set[SegmentKey],
        *,
        attempt: int,
        frame_count: int,
    ) -> tuple[set[SegmentKey], set[SegmentKey]]:
        state = self._state()
        source_frames = filter_records_by_keys(
            self.frames,
            target_keys,
        )
        selections = select_segment_frames(
            source_frames,
            self.scout,
            selected_frame_count=frame_count,
            scout_fps=self.config.scout_fps,
        )
        selected, direct_full = selection_maps(selections)
        attempt_keys = [
            key for key in state.ordered_keys if key in selected
        ]
        if not attempt_keys:
            return set(), direct_full

        detail_frames = materialize_selected_frames(
            source_frames,
            [selected[key] for key in attempt_keys],
            output_dir=self.config.selected_frames_dir,
            jpeg_quality=self.config.jpeg_quality,
        )
        outcome = self._perform_attempt(
            attempt_keys,
            attempt=attempt,
            tier=f"tier{attempt}_selected_{frame_count}",
            frames=detail_frames,
        )
        return state.apply(outcome, selected), direct_full

    def _run_full_tier(self, keys: set[SegmentKey]) -> None:
        state = self._state()
        source_frames = filter_records_by_keys(self.frames, keys)
        selections = {
            key: full_segment_selection(
                records,
                "Using the complete 2 FPS segment after "
                "selection or quality fallback.",
            )
            for key, records in group_records_by_segment(
                source_frames
            ).items()
        }
        ordered_keys = [
            key for key in state.ordered_keys if key in keys
        ]
        outcome = self._perform_attempt(
            ordered_keys,
            attempt=3,
            tier="tier3_full_2fps",
            frames=full_segment_records(source_frames),
        )
        state.apply(outcome, selections)
        state.keep_final_failures(outcome)

    def _perform_attempt(
        self,
        keys: list[SegmentKey],
        *,
        attempt: int,
        tier: str,
        frames: list[dict[str, object]],
    ) -> AttemptOutcome:
        state = self._state()
        if self.ocr is None:
            raise LineupOCRError("OCR model is not initialized.")
        return perform_attempt(
            keys=keys,
            attempt=attempt,
            tier=tier,
            target_records=frames,
            previous_records=state.active_frames,
            previous_detections=state.active_detections,
            ocr=self.ocr,
            config=self.config,
        )

    def _validate_completion(self) -> None:
        state = self._state()
        missing = set(state.ordered_keys) - set(state.final_quality)
        if missing:
            raise LineupOCRError(
                "Pipeline did not produce a final quality result for: "
                + ", ".join(
                    f"{video} segment {index}"
                    for video, index in sorted(missing)
                )
            )

    def _write_outputs(self) -> None:
        state = self._state()

        def flatten(
            grouped: dict[SegmentKey, list[dict[str, object]]],
        ) -> list[dict[str, object]]:
            return [
                row
                for key in state.ordered_keys
                for row in grouped.get(key, [])
            ]

        diagnostics = [
            final_diagnostic(
                key,
                state.final_diagnostics.get(key, {}),
                state.final_quality[key],
                state.final_tier[key],
            )
            for key in state.ordered_keys
        ]
        selection_diagnostics = selection_diagnostic_rows(
            [
                state.final_selection[key]
                for key in state.ordered_keys
            ]
        )
        outputs = (
            (
                flatten(state.active_detections),
                self.config.output_csv,
                DETECTION_COLUMNS,
            ),
            (
                flatten(state.active_frames),
                self.config.selected_frames_csv,
                SELECTED_FRAME_COLUMNS,
            ),
            (
                selection_diagnostics,
                self.config.selection_diagnostics_csv,
                SELECTION_DIAGNOSTIC_COLUMNS,
            ),
            (
                flatten(state.final_records),
                self.config.resolved_output_csv,
                RESOLVED_COLUMNS,
            ),
            (
                diagnostics,
                self.config.resolved_diagnostics_csv,
                DIAGNOSTIC_COLUMNS,
            ),
            (
                state.attempt_rows,
                self.config.attempts_csv,
                ATTEMPT_COLUMNS,
            ),
        )
        for rows, path, columns in outputs:
            write_csv(rows, path, columns)

    def _print_summary(self) -> int:
        state = self._state()
        passed = sum(
            quality.passed
            for quality in state.final_quality.values()
        )
        player_rows = sum(
            len(records)
            for records in state.final_records.values()
        )
        print(
            f"Pipeline complete: {passed}/{len(state.ordered_keys)} "
            f"segment(s) passed the quality gate; "
            f"{player_rows} player rows."
        )
        print(
            f"Resolved lineups: {self.config.resolved_output_csv}"
        )
        print(f"Attempt diagnostics: {self.config.attempts_csv}")
        return 0 if passed == len(state.ordered_keys) else 2

    def _state(self) -> WorkflowState:
        if self.state is None:
            raise LineupOCRError("Workflow state is not initialized.")
        return self.state
