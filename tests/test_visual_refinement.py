from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lineup.schema import LineupDetectionResult, LineupSegment
from lineup.ocr_worker import (
    Detection,
    Recognition,
    Unit,
    _events,
    _middle_out,
    _recognition_score,
    _shortlist,
)
from lineup.visual_refinement import (
    VisualEvent,
    _credible_local_events,
    build_visual_units,
    merge_visual_evidence,
    refine_with_visual_ocr,
    snap_visual_event,
)


def coarse_segment() -> LineupSegment:
    return LineupSegment(
        segment_id="lineup_001",
        team_name="Team A",
        start_seconds=100.0,
        end_seconds=130.0,
        confidence=0.8,
        evidence_segment_ids=("chunk_001",),
        start_anchor_text="starting eleven",
        end_anchor_text="last player",
        reason="The commentator reads Team A's lineup.",
        evidence_start_seconds=60.0,
        evidence_end_seconds=120.0,
    )


class VisualRefinementTests(unittest.TestCase):
    def test_worker_visits_the_coarse_center_before_both_edges(self) -> None:
        units = (
            Unit(0, 0.0, 3.0, 1.5),
            Unit(1, 3.0, 6.0, 4.5),
            Unit(2, 6.0, 9.0, 7.5),
        )

        self.assertEqual(_middle_out(units, 4.5), (1, 0, 2))

    def test_tiny_ocr_gate_rejects_a_normal_scoreboard(self) -> None:
        scoreboard = _recognition_score(
            ("SETANTA SPORTS", "LIVE", "2", "1"),
            Detection(box_count=5, area_ratio=0.01),
        )
        lineup = _recognition_score(
            (
                "4-3-2-1",
                "SUBSTITUTES",
                "1 LEALI",
                "16 BIJLOW",
                "22 VASQUEZ",
                "32 FRASURE",
                "9 VITINHA",
            ),
            Detection(box_count=18, area_ratio=0.08),
        )

        self.assertFalse(scoreboard.positive)
        self.assertTrue(lineup.positive)

        advertising_board = _recognition_score(
            (
                "FIFA",
                "BEL 0-0 JPN 02:53",
                "WANDA",
                "GAZPROM",
                "ROSTOV-ON-DON",
                "3:00",
            ),
            Detection(
                box_count=13,
                area_ratio=0.043,
                x_span=0.93,
                y_span=0.89,
            ),
        )
        self.assertFalse(advertising_board.positive)

    def test_tiny_ocr_accepts_simple_jersey_name_and_formation_signals(self) -> None:
        lineup_layout = Detection(
            box_count=8,
            area_ratio=0.03,
            x_span=0.55,
            y_span=0.6,
        )

        jersey_names = _recognition_score(
            (
                "1 KIM",
                "2 CHO",
                "39 MIN",
                "24 LEE",
                "9 PARK",
                "10 SON",
                "18 HWANG",
                "21 JOE",
            ),
            Detection(
                box_count=18,
                area_ratio=0.12,
                x_span=0.55,
                y_span=0.6,
            ),
        )
        formation = _recognition_score(
            ("4-3-3", "MBAPPE", "KANTE", "SALIBA", "7 DEMBELE"),
            lineup_layout,
        )

        self.assertTrue(jersey_names.positive)
        self.assertTrue(formation.positive)

    def test_event_expands_across_short_scene_and_sparse_transition(self) -> None:
        units = (
            Unit(0, 0.0, 3.0, 1.5, 0),
            Unit(1, 3.0, 6.0, 4.5, 1),
            Unit(2, 6.0, 9.0, 7.5, 1),
            Unit(3, 9.0, 12.0, 10.5, 1),
            Unit(4, 12.0, 15.0, 13.5, 2),
            Unit(5, 15.0, 18.0, 16.5, 3),
        )
        graphic = Detection(
            box_count=9,
            area_ratio=0.03,
            x_span=0.6,
            y_span=0.7,
            layout_cells=(0, 8, 16, 24, 32),
        )
        detections = {
            0: Detection(2, 0.004),
            1: Detection(
                4,
                0.012,
                x_span=0.3,
                y_span=0.5,
                layout_cells=(0, 8, 16),
            ),
            2: graphic,
            3: Detection(0, 0.0),
            4: graphic,
            5: Detection(2, 0.004),
        }
        recognitions = {
            2: Recognition(True, 0.8, ("4-3-3", "1 KIM", "2 CHO")),
            4: Recognition(True, 0.82, ("SUBSTITUTES", "9 LEE", "10 PARK")),
        }

        events = _events(
            "lineup_001",
            units,
            detections,
            recognitions,
            center_seconds=8.0,
            max_events=1,
            prefer_center=True,
        )

        self.assertEqual(
            (events[0]["start_seconds"], events[0]["end_seconds"]),
            (3.0, 15.0),
        )

    def test_event_can_expand_to_start_of_a_long_lineup_scene(self) -> None:
        units = tuple(
            Unit(index, index * 3.0, (index + 1) * 3.0, index * 3.0 + 1.5, 0)
            for index in range(12)
        )
        layout = Detection(
            box_count=18,
            area_ratio=0.1,
            x_span=0.6,
            y_span=0.7,
            layout_cells=(0, 8, 16, 24, 32),
        )
        detections = {index: layout for index in range(len(units))}
        recognitions = {
            9: Recognition(True, 0.8, ("1 KIM", "2 CHO", "39 MIN")),
            10: Recognition(True, 0.8, ("4-4-2", "9 LEE", "10 PARK")),
        }

        events = _events(
            "lineup_001",
            units,
            detections,
            recognitions,
            center_seconds=30.0,
            max_events=1,
            prefer_center=True,
        )

        self.assertEqual(events[0]["start_seconds"], 0.0)

    def test_event_crosses_only_visually_similar_scene_boundaries(self) -> None:
        units = (
            Unit(0, 0.0, 3.0, 1.5, 0),
            Unit(1, 3.0, 6.0, 4.5, 0),
            Unit(2, 6.0, 9.0, 7.5, 1),
            Unit(3, 9.0, 12.0, 10.5, 1),
        )
        common = {
            "box_count": 18,
            "area_ratio": 0.1,
            "x_span": 0.6,
            "y_span": 0.7,
            "layout_cells": (0, 8, 16, 24, 32),
        }
        graphic = Detection(**common, appearance_histogram=(1.0, 0.0))
        live = Detection(**common, appearance_histogram=(0.0, 1.0))
        recognitions = {
            3: Recognition(True, 0.8, ("1 KIM", "2 CHO", "39 MIN")),
        }

        stopped = _events(
            "lineup_001",
            units,
            {0: live, 1: live, 2: graphic, 3: graphic},
            recognitions,
            center_seconds=10.0,
            max_events=1,
            prefer_center=True,
        )
        expanded = _events(
            "lineup_001",
            units,
            {index: graphic for index in range(4)},
            recognitions,
            center_seconds=10.0,
            max_events=1,
            prefer_center=True,
        )
        identity_expanded = _events(
            "lineup_001",
            units,
            {0: live, 1: live, 2: graphic, 3: graphic},
            {
                1: Recognition(False, 0.3, ("TEAM A", "COACH")),
                3: Recognition(True, 0.8, ("TEAM A", "1 KIM", "2 CHO")),
            },
            center_seconds=10.0,
            max_events=1,
            prefer_center=True,
        )

        self.assertEqual(stopped[0]["start_seconds"], 6.0)
        self.assertEqual(expanded[0]["start_seconds"], 0.0)
        self.assertEqual(identity_expanded[0]["start_seconds"], 0.0)

    def test_event_does_not_cross_into_long_live_scene_before_graphic(self) -> None:
        units = tuple(
            Unit(index, index * 3.0, (index + 1) * 3.0, index * 3.0 + 1.5, 0)
            for index in range(7)
        ) + tuple(
            Unit(index, index * 3.0, (index + 1) * 3.0, index * 3.0 + 1.5, 1)
            for index in range(7, 11)
        )
        layout = Detection(
            box_count=9,
            area_ratio=0.03,
            x_span=0.6,
            y_span=0.7,
            layout_cells=(0, 8, 16, 24),
        )
        detections = {index: layout for index in range(len(units))}
        recognitions = {
            8: Recognition(True, 0.8, ("1 KIM", "2 CHO", "39 MIN")),
            10: Recognition(True, 0.8, ("4-4-2", "9 LEE", "10 PARK")),
        }

        events = _events(
            "lineup_001",
            units,
            detections,
            recognitions,
            center_seconds=27.0,
            max_events=1,
            prefer_center=True,
        )

        self.assertEqual(events[0]["start_seconds"], 21.0)

    def test_visual_snap_only_expands_outward(self) -> None:
        event = VisualEvent(133.152, 164.12, 0.9, "lineup_001")

        snapped = snap_visual_event(
            event,
            (136.0, 138.0, 164.12),
            radius_seconds=4.0,
        )

        self.assertEqual(snapped.start_seconds, 133.152)
        self.assertEqual(snapped.end_seconds, 164.12)

        internal_slide = snap_visual_event(
            VisualEvent(282.08, 305.28, 0.9, "lineup_002"),
            (279.8, 282.08, 305.28),
            radius_seconds=4.0,
        )
        self.assertEqual(internal_slide.start_seconds, 279.8)
        self.assertEqual(internal_slide.end_seconds, 305.28)

    def test_low_confidence_local_event_must_match_the_qwen_team(self) -> None:
        false_scoreboard = VisualEvent(
            353.8,
            362.834,
            0.55,
            "lineup_001",
            texts=("FIFA", "BEL 0-0 JPN", "WANDA"),
        )
        belgium = VisualEvent(
            0.0,
            34.733,
            0.55,
            "lineup_001",
            texts=("BELGIUM", "COURTOIS"),
        )

        self.assertEqual(
            _credible_local_events((false_scoreboard,), "Belgium"),
            (),
        )
        self.assertEqual(
            _credible_local_events((belgium,), "Belgium"),
            (belgium,),
        )

    def test_global_shortlist_preserves_temporal_coverage(self) -> None:
        units = tuple(
            Unit(index, float(index), float(index + 1), index + 0.5)
            for index in range(14)
        )
        detections = {
            index: Detection(
                box_count=20,
                area_ratio=0.12 if index < 11 else 0.08,
            )
            for index in range(14)
        }

        selected = _shortlist(
            detections,
            units=units,
            center_order=tuple(range(14)),
            limit=6,
        )

        self.assertLessEqual(len(selected), 6)
        self.assertTrue(any(index >= 12 for index in selected))

    def test_splits_scenes_only_enough_to_cover_overlay_changes(self) -> None:
        units = build_visual_units(
            start_seconds=0.0,
            end_seconds=10.0,
            scene_cuts=(4.0, 9.0),
            max_unit_seconds=3.0,
        )

        self.assertEqual(
            [(unit.start_seconds, unit.end_seconds) for unit in units],
            [(0.0, 2.0), (2.0, 4.0), (4.0, 6.5), (6.5, 9.0), (9.0, 10.0)],
        )

    def test_local_ocr_replaces_timestamps_but_keeps_qwen_identity(self) -> None:
        result = LineupDetectionResult(
            model="qwen-plus",
            segments=(coarse_segment(),),
            raw_response={"lineup_segments": []},
            status="complete",
        )
        event = VisualEvent(90.0, 118.0, 0.91, "lineup_001")

        refined = merge_visual_evidence(
            result,
            local_events={"lineup_001": (event,)},
            fallback_events=(),
            expected_count=1,
            diagnostics={"fallback_used": False},
        )

        self.assertEqual(refined.segments[0].team_name, "Team A")
        self.assertEqual(refined.segments[0].start_seconds, 90.0)
        self.assertEqual(refined.segments[0].end_seconds, 118.0)
        self.assertEqual(refined.status, "complete")

    def test_global_fallback_can_recover_a_lineup_without_commentary(self) -> None:
        result = LineupDetectionResult(
            model="qwen-plus",
            segments=(),
            raw_response={"lineup_segments": []},
            status="empty",
        )
        event = VisualEvent(180.0, 210.0, 0.87, "global_fallback")

        refined = merge_visual_evidence(
            result,
            local_events={},
            fallback_events=(event,),
            expected_count=1,
            diagnostics={"fallback_used": True},
        )

        self.assertEqual(len(refined.segments), 1)
        self.assertEqual(refined.segments[0].team_name, "Visual lineup 1")
        self.assertEqual(refined.status, "complete")
        self.assertIn(
            "visual_lineup_has_no_transcript_team_name",
            refined.raw_response["_review_reasons"],
        )

    def test_fallback_matches_team_text_and_does_not_reuse_local_event(self) -> None:
        japan = LineupSegment(
            segment_id="lineup_001",
            team_name="Japan",
            start_seconds=60.0,
            end_seconds=120.0,
            confidence=0.95,
            evidence_segment_ids=("chunk_001",),
            start_anchor_text="Japan starting lineup",
            end_anchor_text="Japan final player",
            reason="The commentator reads Japan's lineup.",
        )
        belgium = LineupSegment(
            segment_id="lineup_002",
            team_name="Belgium",
            start_seconds=360.0,
            end_seconds=420.0,
            confidence=0.9,
            evidence_segment_ids=("chunk_006",),
            start_anchor_text="Belgium starting lineup",
            end_anchor_text="Belgium final player",
            reason="The commentator reads Belgium's lineup late.",
        )
        result = LineupDetectionResult(
            model="qwen-plus",
            segments=(japan, belgium),
            raw_response={"lineup_segments": []},
            status="complete",
        )
        japan_event = VisualEvent(
            67.7,
            102.1,
            0.8,
            "lineup_001",
            texts=("JAPAN", "Eiji KAWASHIMA"),
        )
        belgium_event = VisualEvent(
            2.067,
            34.733,
            0.79,
            "global_fallback",
            texts=("fifa.com", "BELGIUM", "Substitutes"),
        )

        refined = merge_visual_evidence(
            result,
            local_events={"lineup_001": (japan_event,)},
            fallback_events=(belgium_event, japan_event),
            expected_count=2,
            diagnostics={"fallback_used": True},
        )

        self.assertEqual(
            [
                (segment.team_name, segment.start_seconds)
                for segment in refined.segments
            ],
            [("Belgium", 2.067), ("Japan", 67.7)],
        )

    def test_fallback_reads_team_title_for_lineup_without_transcript(self) -> None:
        chelsea = LineupSegment(
            segment_id="lineup_001",
            team_name="Chelsea",
            start_seconds=420.0,
            end_seconds=480.0,
            confidence=0.95,
            evidence_segment_ids=("chunk_007",),
            start_anchor_text="Chelsea starting lineup",
            end_anchor_text="Chelsea final player",
            reason="The commentator reads Chelsea's lineup late.",
        )
        result = LineupDetectionResult(
            model="qwen-plus",
            segments=(chelsea,),
            raw_response={"lineup_segments": []},
            status="complete",
        )

        refined = merge_visual_evidence(
            result,
            local_events={},
            fallback_events=(
                VisualEvent(
                    260.0,
                    292.56,
                    0.79,
                    "global_fallback",
                    texts=("NOAH", "CANCAREVIC"),
                ),
                VisualEvent(
                    136.0,
                    164.12,
                    0.78,
                    "global_fallback",
                    texts=("CHELSEA FC", "SUBSTITUTES"),
                ),
            ),
            expected_count=2,
            diagnostics={"fallback_used": True},
        )

        self.assertEqual(
            [
                (segment.team_name, segment.start_seconds)
                for segment in refined.segments
            ],
            [("Chelsea", 136.0), ("NOAH", 260.0)],
        )
        self.assertIn(
            "visual_lineup_has_no_transcript_evidence",
            refined.raw_response["_review_reasons"],
        )

    def test_short_visual_event_requires_boundary_review(self) -> None:
        result = LineupDetectionResult(
            model="qwen-plus",
            segments=(coarse_segment(),),
            raw_response={"lineup_segments": []},
            status="complete",
        )

        refined = merge_visual_evidence(
            result,
            local_events={
                "lineup_001": (
                    VisualEvent(228.84, 233.4, 0.91, "lineup_001"),
                )
            },
            fallback_events=(),
            expected_count=1,
            diagnostics={"fallback_used": False},
        )

        self.assertIn(
            "visual_lineup_boundary_may_be_incomplete",
            refined.raw_response["_review_reasons"],
        )

    def test_successful_local_search_does_not_run_global_fallback(self) -> None:
        calls: list[str] = []

        def fake_scene_detector(*_args: object, **_kwargs: object) -> tuple[float, ...]:
            calls.append("scene")
            return (80.0, 120.0, 140.0)

        def fake_ocr_runner(
            _video: Path, tasks: object, **_kwargs: object
        ) -> dict[str, object]:
            calls.append("ocr")
            task = tuple(tasks)[0]
            return {
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "events": [
                            {
                                "start_seconds": 80.0,
                                "end_seconds": 120.0,
                                "confidence": 0.9,
                            }
                        ],
                    }
                ]
            }

        result = LineupDetectionResult(
            model="qwen-plus",
            segments=(coarse_segment(),),
            raw_response={"lineup_segments": []},
            status="complete",
        )
        refined = refine_with_visual_ocr(
            result,
            video_path=Path("unused.mp4"),
            scan_end_seconds=900.0,
            expected_count=1,
            scene_threshold=27.0,
            scene_min_length_frames=None,
            scene_min_length_seconds=0.5,
            scene_snap_radius_seconds=4.0,
            scene_detector=fake_scene_detector,
            ocr_runner=fake_ocr_runner,
        )

        self.assertEqual(calls, ["scene", "ocr"])
        self.assertFalse(refined.raw_response["visual_refinement"]["fallback_used"])

    def test_scene_detection_skips_gaps_between_distant_local_windows(self) -> None:
        scene_ranges: list[tuple[float, float]] = []

        def fake_scene_detector(
            _video: Path,
            *,
            start_seconds: float,
            end_seconds: float,
            **_kwargs: object,
        ) -> tuple[float, ...]:
            scene_ranges.append((start_seconds, end_seconds))
            return (start_seconds + 1.0, end_seconds - 1.0)

        def fake_ocr_runner(
            _video: Path, tasks: object, **_kwargs: object
        ) -> dict[str, object]:
            return {
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "events": [
                            {
                                "start_seconds": task.units[0].start_seconds,
                                "end_seconds": task.units[-1].end_seconds,
                                "confidence": 0.9,
                            }
                        ],
                    }
                    for task in tasks
                ]
            }

        second = LineupSegment(
            segment_id="lineup_002",
            team_name="Team B",
            start_seconds=400.0,
            end_seconds=430.0,
            confidence=0.8,
            evidence_segment_ids=("chunk_006",),
            start_anchor_text="Team B starting eleven",
            end_anchor_text="Team B last player",
            reason="The commentator reads Team B's lineup.",
            evidence_start_seconds=360.0,
            evidence_end_seconds=420.0,
        )
        result = LineupDetectionResult(
            model="qwen-plus",
            segments=(coarse_segment(), second),
            raw_response={"lineup_segments": []},
            status="complete",
        )

        refined = refine_with_visual_ocr(
            result,
            video_path=Path("unused.mp4"),
            scan_end_seconds=600.0,
            expected_count=2,
            scene_threshold=27.0,
            scene_min_length_frames=None,
            scene_min_length_seconds=0.5,
            scene_snap_radius_seconds=4.0,
            scene_detector=fake_scene_detector,
            ocr_runner=fake_ocr_runner,
        )

        self.assertEqual(scene_ranges, [(48.0, 142.0), (348.0, 442.0)])
        self.assertFalse(refined.raw_response["visual_refinement"]["fallback_used"])

    def test_empty_qwen_result_uses_sparse_fallback_capped_at_600s(self) -> None:
        scene_ranges: list[tuple[float, float]] = []

        def fake_scene_detector(
            _video: Path, *, start_seconds: float, end_seconds: float, **_kwargs: object
        ) -> tuple[float, ...]:
            scene_ranges.append((start_seconds, end_seconds))
            return (200.0, 230.0)

        def fake_ocr_runner(
            _video: Path, tasks: object, **_kwargs: object
        ) -> dict[str, object]:
            task = tuple(tasks)[0]
            return {
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "events": [
                            {
                                "start_seconds": 200.0,
                                "end_seconds": 230.0,
                                "confidence": 0.9,
                            }
                        ],
                    }
                ]
            }

        result = LineupDetectionResult(
            model="qwen-plus",
            segments=(),
            raw_response={"lineup_segments": []},
            status="empty",
        )
        refined = refine_with_visual_ocr(
            result,
            video_path=Path("unused.mp4"),
            scan_end_seconds=900.0,
            expected_count=1,
            scene_threshold=27.0,
            scene_min_length_frames=None,
            scene_min_length_seconds=0.5,
            scene_snap_radius_seconds=4.0,
            scene_detector=fake_scene_detector,
            ocr_runner=fake_ocr_runner,
        )

        self.assertEqual(scene_ranges, [(0.0, 600.0)])
        self.assertEqual(len(refined.segments), 1)
        self.assertTrue(refined.raw_response["visual_refinement"]["fallback_used"])


if __name__ == "__main__":
    unittest.main()
