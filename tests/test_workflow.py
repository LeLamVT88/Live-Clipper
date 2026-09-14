from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mapping.pipeline import (
    AttemptOutcome,
    lineups_expected_in_segment,
    reusable_attempt_data,
)
from mapping.config import DEFAULTS, PipelineConfig, build_parser
from mapping.resolver.quality import (
    QualityResult,
    evaluate_segment_quality,
    select_match_lineups,
)
from mapping.selector import SegmentSelection
from mapping.pipeline import LineupWorkflow, WorkflowState
import mapping.pipeline as workflow


def resolved_players() -> list[dict[str, object]]:
    return [
        {
            "video": "match.mp4",
            "segment_index": 1,
            "lineup_index": 1,
            "slot_index": index,
            "shirt_number": index,
            "player_name": f"PLAYER {chr(64 + index)}",
            "number_source": "detector",
            "number_evidence_frames": 2,
            "pair_confidence": 0.95,
        }
        for index in range(1, 12)
    ]


def frame(frame_index: int) -> dict[str, object]:
    return {
        "video": "match.mp4",
        "segment_index": 1,
        "segment_label": "lineup_01",
        "segment_start_seconds": 0.0,
        "segment_end_seconds": 4.0,
        "frame_index": frame_index,
        "frame_path": f"frame_{frame_index}.jpg",
        "timestamp": f"00:00:0{(frame_index - 1) / 2:g}",
        "timestamp_seconds": (frame_index - 1) / 2,
        "relative_seconds": (frame_index - 1) / 2,
        "frame_width": 1920,
        "frame_height": 1080,
    }


class QualityGateTests(unittest.TestCase):
    def test_accepts_complete_unique_confident_lineup(self) -> None:
        result = evaluate_segment_quality(
            resolved_players(),
            {"status": "resolved"},
            expected_players=11,
            min_pair_confidence=0.80,
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.resolved_players, 11)

    def test_rejects_missing_or_duplicate_shirt_numbers(self) -> None:
        duplicate = resolved_players()
        duplicate[-1]["shirt_number"] = 1

        duplicate_result = evaluate_segment_quality(
            duplicate,
            {"status": "resolved"},
            expected_players=11,
            min_pair_confidence=0.80,
        )
        missing_result = evaluate_segment_quality(
            resolved_players()[:-1],
            {"status": "resolved"},
            expected_players=11,
            min_pair_confidence=0.80,
        )

        self.assertFalse(duplicate_result.passed)
        self.assertIn("10/11 unique", duplicate_result.message)
        self.assertFalse(missing_result.passed)
        self.assertIn("10/11 players", missing_result.message)

    def test_rejects_ui_name_empty_name_and_bad_confidence(self) -> None:
        records = resolved_players()
        records[0]["player_name"] = "TEAM FORMATION"
        records[1]["player_name"] = ""
        records[2]["pair_confidence"] = math.nan

        result = evaluate_segment_quality(
            records,
            {"status": "resolved"},
            expected_players=11,
            min_pair_confidence=0.80,
        )

        self.assertFalse(result.passed)
        self.assertIn("UI text", result.message)
        self.assertIn("empty player name", result.message)
        self.assertIn("below confidence", result.message)

    def test_rejects_non_resolved_diagnostic(self) -> None:
        result = evaluate_segment_quality(
            resolved_players(),
            {"status": "unresolved"},
            expected_players=11,
            min_pair_confidence=0.80,
        )

        self.assertFalse(result.passed)
        self.assertIn("resolver status", result.message)

    def test_rejects_single_frame_local_ocr_shirt_number(self) -> None:
        records = resolved_players()
        records[0]["number_source"] = "local_ocr"
        records[0]["number_evidence_frames"] = 1

        result = evaluate_segment_quality(
            records,
            {"status": "resolved"},
            expected_players=11,
            min_pair_confidence=0.80,
        )

        self.assertFalse(result.passed)
        self.assertIn("fewer than 2 frames", result.message)

    def test_accepts_single_frame_local_number_with_scout_confirmed_name(self) -> None:
        records = resolved_players()
        records[0].update(
            {
                "number_source": "local_ocr",
                "number_evidence_frames": 1,
                "name_source": "scout",
                "full_name_evidence_frames": 3,
            }
        )

        result = evaluate_segment_quality(
            records,
            {"status": "resolved"},
            expected_players=11,
            min_pair_confidence=0.80,
        )

        self.assertTrue(result.passed)

    def test_single_clip_requires_both_lineups(self) -> None:
        one_lineup = resolved_players()
        second_lineup = [
            dict(row, lineup_index=2, player_name=f"OPPONENT {index}")
            for index, row in enumerate(one_lineup, start=1)
        ]

        incomplete = evaluate_segment_quality(
            one_lineup, {"status": "resolved"}, 11, 0.80, expected_lineups=2
        )
        complete = evaluate_segment_quality(
            one_lineup + second_lineup,
            {"status": "resolved"}, 11, 0.80, expected_lineups=2,
        )

        self.assertFalse(incomplete.passed)
        self.assertIn("1/2 expected lineups", incomplete.message)
        self.assertTrue(complete.passed)

    def test_match_gate_rejects_duplicate_team_from_two_clips(self) -> None:
        first = resolved_players()
        duplicate = [dict(row, video="duplicate.mp4") for row in first]

        selected, quality = select_match_lineups(first + duplicate, 2, 11)

        self.assertEqual(selected, [])
        self.assertFalse(quality.passed)
        self.assertIn("1/2 distinct", quality.message)

    def test_match_gate_keeps_two_distinct_teams(self) -> None:
        first = resolved_players()
        second = [
            dict(row, video="opponent.mp4", player_name=f"OPPONENT {index}")
            for index, row in enumerate(first, start=1)
        ]

        selected, quality = select_match_lineups(first + second, 2, 11)

        self.assertTrue(quality.passed)
        self.assertEqual(len(selected), 22)
        self.assertEqual({row["lineup_index"] for row in selected}, {1, 2})


class PipelineConfigTests(unittest.TestCase):
    def test_parser_exposes_every_configured_path(self) -> None:
        args = build_parser().parse_args([])

        for name in DEFAULTS:
            self.assertTrue(hasattr(args, name), name)

    def test_match_lineup_count_is_configurable(self) -> None:
        args = build_parser().parse_args(["--lineups-per-match", "1"])

        self.assertEqual(args.lineups_per_match, 1)


class AdaptiveFallbackTests(unittest.TestCase):
    def test_two_clips_keep_one_lineup_fast_path_per_segment(self) -> None:
        self.assertEqual(lineups_expected_in_segment(1, 2), 2)
        self.assertEqual(lineups_expected_in_segment(2, 2), 1)

    def test_reuses_prior_frames_and_only_ocr_new_expansion(self) -> None:
        previous_records = [frame(index) for index in range(1, 4)]
        target_records = [frame(index) for index in range(1, 8)]
        previous_detections = [
            {**frame(1), "text": "ONE"},
            {**frame(2), "text": "TWO"},
        ]

        new_records, reused_detections = reusable_attempt_data(
            target_records,
            previous_records,
            previous_detections,
        )

        self.assertEqual(
            [record["frame_index"] for record in new_records],
            [4, 5, 6, 7],
        )
        self.assertEqual(
            [record["frame_index"] for record in reused_detections],
            [1, 2],
        )

    def test_accepts_only_segments_that_pass_quality_gate(self) -> None:
        passed_key = ("match.mp4", 1)
        failed_key = ("match.mp4", 2)
        passed_quality = QualityResult(True, 11, "passed")
        failed_quality = QualityResult(False, 10, "missing player")
        outcome = AttemptOutcome(
            tier="tier1_selected_3",
            frame_records={passed_key: [], failed_key: []},
            detections={passed_key: [], failed_key: []},
            resolved_records={
                passed_key: resolved_players(),
                failed_key: resolved_players()[:-1],
            },
            diagnostics={
                passed_key: {"status": "resolved"},
                failed_key: {"status": "unresolved"},
            },
            quality={
                passed_key: passed_quality,
                failed_key: failed_quality,
            },
            attempt_rows=[],
        )
        state = WorkflowState([passed_key, failed_key])
        selection = SegmentSelection(
            video=passed_key[0],
            segment_index=passed_key[1],
            layout="formation",
            status="selected",
            score=50.0,
            scout_frame_index=1,
            scout_timestamp_seconds=0.0,
            formation_anchor_count=11,
            table_pair_count=0,
            number_count=11,
            name_count=11,
            crop_x1_norm=0.0,
            crop_x2_norm=1.0,
            selected_frame_indices=(1, 2, 3),
            message="selected",
        )

        failed = state.apply(
            outcome,
            {passed_key: selection},
        )

        self.assertEqual(failed, {failed_key})
        self.assertIn(passed_key, state.final_records)
        self.assertNotIn(failed_key, state.final_records)
        self.assertEqual(
            state.final_tier[passed_key],
            "tier1_selected_3",
        )

    def test_failed_attempt_keeps_selection_for_final_diagnostics(self) -> None:
        key = ("match.mp4", 1)
        quality = QualityResult(False, 0, "still unresolved")
        outcome = AttemptOutcome(
            tier="tier3_full_2fps",
            frame_records={key: [frame(1)]},
            detections={key: []},
            resolved_records={key: []},
            diagnostics={
                key: {
                    "video": key[0],
                    "segment_index": key[1],
                    "status": "unresolved",
                    "resolution_method": "",
                    "resolved_players": 0,
                    "message": "no complete lineup",
                }
            },
            quality={key: quality},
            attempt_rows=[],
        )
        selection = SegmentSelection(
            video=key[0],
            segment_index=key[1],
            layout="fallback_full_segment",
            status="fallback",
            score=0.0,
            scout_frame_index=0,
            scout_timestamp_seconds=0.0,
            formation_anchor_count=0,
            table_pair_count=0,
            number_count=0,
            name_count=0,
            crop_x1_norm=0.0,
            crop_x2_norm=1.0,
            selected_frame_indices=(1,),
            message="full segment fallback",
        )
        state = WorkflowState([key])

        failed = state.apply(outcome, {key: selection}, final=True)

        self.assertEqual(failed, {key})
        self.assertEqual(state.final_selection[key], selection)
        self.assertEqual(state.final_quality[key], quality)

    def test_workflow_runs_three_seven_then_full_segment(self) -> None:
        key = ("match.mp4", 1)
        frames = [frame(index) for index in range(1, 9)]
        attempts: list[int] = []

        def selection(
            frame_records: list[dict[str, object]],
            scout_detections: object,
            selected_frame_count: int,
            scout_fps: float,
            expected_lineups: int,
        ) -> list[SegmentSelection]:
            del frame_records, scout_detections, scout_fps, expected_lineups
            return [
                SegmentSelection(
                    video=key[0],
                    segment_index=key[1],
                    layout="formation",
                    status="selected",
                    score=50.0,
                    scout_frame_index=1,
                    scout_timestamp_seconds=0.0,
                    formation_anchor_count=11,
                    table_pair_count=0,
                    number_count=11,
                    name_count=11,
                    crop_x1_norm=0.0,
                    crop_x2_norm=1.0,
                    selected_frame_indices=tuple(range(1, selected_frame_count + 1)),
                    message="selected",
                )
            ]

        def materialize(
            frame_records: list[dict[str, object]],
            selections: list[SegmentSelection],
            output_dir: Path,
            jpeg_quality: int,
        ) -> list[dict[str, object]]:
            del output_dir, jpeg_quality
            selected = set(selections[0].selected_frame_indices)
            return [record for record in frame_records if int(record["frame_index"]) in selected]

        def attempt(**kwargs: object) -> AttemptOutcome:
            attempt_number = int(kwargs["attempt"])
            attempts.append(attempt_number)
            target_records = list(kwargs["target_records"])
            tier = str(kwargs["tier"])
            passed = attempt_number == 3
            quality = QualityResult(
                passed=passed,
                resolved_players=11 if passed else 0,
                message="passed" if passed else "retry",
            )
            return AttemptOutcome(
                tier=tier,
                frame_records={key: target_records},
                detections={key: []},
                resolved_records={key: resolved_players() if passed else []},
                diagnostics={
                    key: {
                        "video": key[0],
                        "segment_index": key[1],
                        "status": ("resolved" if passed else "unresolved"),
                        "resolution_method": ("formation" if passed else ""),
                        "resolved_players": (11 if passed else 0),
                        "message": "",
                    }
                },
                quality={key: quality},
                attempt_rows=[
                    {
                        "video": key[0],
                        "segment_index": key[1],
                        "attempt": attempt_number,
                        "tier": tier,
                    }
                ],
            )

        temporary = Path("/tmp/pipeline-test")
        config = PipelineConfig(
            clips_dir=temporary,
            frames_dir=temporary,
            selected_frames_dir=temporary,
            frames_csv=temporary / "frames.csv",
            scout_output_csv=temporary / "scout.csv",
            selected_frames_csv=temporary / "selected.csv",
            selection_diagnostics_csv=temporary / "selection.csv",
            output_csv=temporary / "raw.csv",
            resolved_output_csv=temporary / "resolved.csv",
            resolved_diagnostics_csv=temporary / "diagnostics.csv",
            attempts_csv=temporary / "attempts.csv",
            cache_dir=temporary,
            fps=2.0,
            scout_fps=0.5,
            jpeg_quality=95,
            min_score=0.80,
            ocr_batch_size=8,
            initial_frame_count=3,
            expanded_frame_count=7,
            players_per_lineup=11,
            lineups_per_match=1,
            min_number_count=8,
            same_lineup_gap_seconds=20.0,
            signature_threshold=0.45,
            min_pair_confidence=0.80,
            disable_local_ocr=False,
            extract_only=False,
        )
        segment = SimpleNamespace(
            video_path=Path("match.mp4"),
            index=1,
            start_seconds=0.0,
            end_seconds=4.0,
        )

        with (
            patch.object(
                workflow,
                "load_lineup_clips",
                return_value=[segment],
            ),
            patch.object(
                workflow,
                "extract_segment_frames",
                return_value=frames,
            ),
            patch.object(
                workflow,
                "sample_scout_frames",
                return_value=[frames[0]],
            ),
            patch.object(workflow, "create_ocr", return_value=object()),
            patch.object(workflow, "run_ocr", return_value=[]),
            patch.object(
                workflow,
                "select_segment_frames",
                side_effect=selection,
            ),
            patch.object(
                workflow,
                "materialize_selected_frames",
                side_effect=materialize,
            ),
            patch.object(
                workflow,
                "perform_attempt",
                side_effect=attempt,
            ) as perform,
            patch.object(workflow, "write_csv"),
        ):
            exit_code = LineupWorkflow(config).run()

        self.assertEqual(exit_code, 0)
        self.assertEqual(attempts, [1, 2, 3])
        self.assertEqual(
            [len(call.kwargs["target_records"]) for call in perform.call_args_list],
            [3, 7, 8],
        )


if __name__ == "__main__":
    unittest.main()
