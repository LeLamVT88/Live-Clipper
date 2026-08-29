from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lineup.scene_detection import (
    _soft_transition_cuts,
    snap_lineup_result,
    snap_timestamp,
)
from lineup.schema import LineupDetectionResult, LineupSegment


class SceneDetectionTests(unittest.TestCase):
    def test_detects_a_gentle_overlay_after_a_static_shot(self) -> None:
        class FakeStatsManager:
            @staticmethod
            def get_metrics(
                frame_number: int,
                _metric_names: list[str],
            ) -> list[float]:
                if frame_number < 60:
                    return [0.2]
                transition = (1.4, 1.8, 3.4, 2.2, 1.6)
                return [transition[(frame_number - 60) % len(transition)]]

        cuts = _soft_transition_cuts(
            FakeStatsManager(),
            start_seconds=0.0,
            end_seconds=4.0,
            frame_rate=30.0,
        )

        self.assertTrue(any(abs(cut - 2.0) < 0.1 for cut in cuts))

    def test_detects_a_moderate_cut_that_settles_into_a_static_shot(self) -> None:
        class FakeStatsManager:
            @staticmethod
            def get_metrics(
                frame_number: int,
                _metric_names: list[str],
            ) -> list[float]:
                if frame_number < 20:
                    return [8.0]
                if frame_number == 20:
                    return [16.0]
                return [1.0]

        cuts = _soft_transition_cuts(
            FakeStatsManager(),
            start_seconds=0.0,
            end_seconds=4.0,
            frame_rate=10.0,
        )

        self.assertIn(2.0, cuts)

    def test_scene_snap_is_narrow_and_never_chases_a_distant_cut(self) -> None:
        nearby = snap_timestamp(568.2, (543.0, 568.834, 580.2), radius_seconds=3)
        distant = snap_timestamp(575.0, (568.834, 580.2), radius_seconds=3)

        self.assertTrue(nearby.applied)
        self.assertEqual(nearby.snapped_seconds, 568.834)
        self.assertFalse(distant.applied)
        self.assertEqual(distant.snapped_seconds, 575.0)

    def test_scene_snap_prefers_graphic_onset_and_exit_directions(self) -> None:
        cuts = (539.767, 541.967, 568.833, 570.4)

        start = snap_timestamp(540, cuts, radius_seconds=4, preference="forward")
        end = snap_timestamp(570, cuts, radius_seconds=4, preference="backward")

        self.assertEqual(start.snapped_seconds, 541.967)
        self.assertEqual(end.snapped_seconds, 568.833)

    def test_scene_pair_ignores_commentary_that_outlasts_graphic(self) -> None:
        lineup = LineupSegment(
            segment_id="lineup_001",
            team_name="Buriram United",
            start_seconds=540,
            end_seconds=580,
            confidence=0.92,
            evidence_segment_ids=("coarse_1",),
            start_anchor_text="Buriram United have made five changes",
            end_anchor_text="Goran Karacic",
            reason="Qwen coarse transcript anchors.",
        )
        result = LineupDetectionResult(
            model="Qwen/Qwen2.5-VL-3B-Instruct",
            segments=(lineup,),
            raw_response={},
            status="complete",
        )

        snapped = snap_lineup_result(
            result,
            (539.767, 541.967, 568.833, 570.4, 580.2),
        )

        self.assertEqual(snapped.segments[0].start_seconds, 541.967)
        self.assertEqual(snapped.segments[0].end_seconds, 568.833)
        self.assertEqual(
            snapped.raw_response["scene_refinement"]["segments"][0][
                "end_strategy"
            ],
            "paired_graphic_block",
        )

    def test_scene_snap_preserves_coarse_text_evidence(self) -> None:
        lineup = LineupSegment(
            segment_id="lineup_001",
            team_name="Buriram United",
            start_seconds=542.2,
            end_seconds=568.2,
            confidence=0.92,
            evidence_segment_ids=("coarse_1",),
            start_anchor_text="Buriram United have made five changes",
            end_anchor_text="Goran Causic",
            reason="Qwen coarse transcript anchors.",
            evidence_start_seconds=540,
            evidence_end_seconds=600,
        )
        result = LineupDetectionResult(
            model="Qwen/Qwen2.5-VL-3B-Instruct",
            segments=(lineup,),
            raw_response={},
            status="complete",
        )

        snapped = snap_lineup_result(
            result,
            (543.034, 568.834, 580.2),
            radius_seconds=3,
        )

        self.assertEqual(snapped.segments[0].start_seconds, 543.034)
        self.assertEqual(snapped.segments[0].end_seconds, 568.834)
        self.assertEqual(snapped.segments[0].evidence_end_seconds, 600)
        scene_data = snapped.raw_response["scene_refinement"]
        self.assertEqual(
            scene_data["segments"][0]["end"]["original_seconds"],
            568.2,
        )

    def test_selects_long_stable_graphic_after_audio_begins(self) -> None:
        lineup = LineupSegment(
            segment_id="lineup_001",
            team_name="Inter",
            start_seconds=412.915,
            end_seconds=456.694,
            confidence=0.97,
            evidence_segment_ids=("chunk_006", "chunk_007"),
            start_anchor_text="six changes today",
            end_anchor_text="starts today",
            reason="Qwen coarse transcript anchors.",
        )
        result = LineupDetectionResult(
            model="qwen-plus",
            segments=(lineup,),
            raw_response={},
            status="complete",
        )

        snapped = snap_lineup_result(
            result,
            (420.342, 429.302, 455.003, 461.443),
        )

        self.assertEqual(snapped.segments[0].start_seconds, 429.302)
        self.assertEqual(snapped.segments[0].end_seconds, 455.003)

    def test_selects_long_stable_graphic_that_outlives_audio(self) -> None:
        lineup = LineupSegment(
            segment_id="lineup_001",
            team_name="Argentina",
            start_seconds=76.757,
            end_seconds=104.865,
            confidence=0.97,
            evidence_segment_ids=("chunk_001",),
            start_anchor_text="Argentina make only one change",
            end_anchor_text="and Marcos Rojo",
            reason="Qwen coarse transcript anchors.",
        )
        result = LineupDetectionResult(
            model="qwen-plus",
            segments=(lineup,),
            raw_response={},
            status="complete",
        )

        snapped = snap_lineup_result(
            result,
            (45.1, 70.4, 82.6, 122.133, 124.6),
        )

        self.assertEqual(snapped.segments[0].start_seconds, 82.6)
        self.assertEqual(snapped.segments[0].end_seconds, 122.133)

    def test_allows_graphic_to_end_well_before_roster_commentary(self) -> None:
        lineup = LineupSegment(
            segment_id="lineup_001",
            team_name="Al-Sadd",
            start_seconds=540.977,
            end_seconds=595.047,
            confidence=0.97,
            evidence_segment_ids=("chunk_009",),
            start_anchor_text="Al-Sadd have not made any changes",
            end_anchor_text="other attacking options available",
            reason="Qwen coarse transcript anchors.",
        )
        result = LineupDetectionResult(
            model="qwen-plus",
            segments=(lineup,),
            raw_response={},
            status="complete",
        )

        snapped = snap_lineup_result(
            result,
            (535.132, 575.031, 589.331, 594.633),
        )

        self.assertEqual(snapped.segments[0].start_seconds, 535.132)
        self.assertEqual(snapped.segments[0].end_seconds, 575.031)
        self.assertEqual(
            snapped.raw_response["scene_refinement"]["segments"][0][
                "end_strategy"
            ],
            "paired_graphic_block",
        )

    def test_auto_detects_two_consecutive_multislide_graphics(self) -> None:
        first = LineupSegment(
            segment_id="lineup_001",
            team_name="Liverpool",
            start_seconds=263.795,
            end_seconds=284.036,
            confidence=0.98,
            evidence_segment_ids=("chunk_004",),
            start_anchor_text="Liverpool goalkeeper",
            end_anchor_text="under the striker",
            reason="Qwen coarse transcript anchors.",
        )
        second = replace(
            first,
            segment_id="lineup_002",
            team_name="Brentford",
            start_seconds=288.494,
            end_seconds=299.94,
            confidence=0.95,
            start_anchor_text="Brentford goalkeeper",
            end_anchor_text="defensive line",
        )
        result = LineupDetectionResult(
            model="qwen-plus",
            segments=(first, second),
            raw_response={},
            status="complete",
        )
        cuts = (
            240.047,
            243.247,
            246.968,
            252.489,
            259.971,
            266.372,
            273.373,
            278.934,
            284.896,
            287.176,
            290.337,
            298.418,
            303.899,
            310.581,
            317.262,
            322.903,
            327.504,
            330.745,
            332.905,
            334.905,
        )

        snapped = snap_lineup_result(result, cuts)

        self.assertEqual(
            (snapped.segments[0].start_seconds, snapped.segments[0].end_seconds),
            (240.047, 284.896),
        )
        self.assertEqual(
            (snapped.segments[1].start_seconds, snapped.segments[1].end_seconds),
            (287.176, 332.905),
        )
        self.assertEqual(
            snapped.raw_response["scene_refinement"]["mode"],
            "consecutive_slideshow_graphics",
        )

    def test_slideshow_keeps_precise_audio_end_when_a_cut_is_nearby(self) -> None:
        first = LineupSegment(
            segment_id="lineup_001",
            team_name="Al-Nassr",
            start_seconds=480.0,
            end_seconds=505.132,
            confidence=0.98,
            evidence_segment_ids=("chunk_008",),
            start_anchor_text="Al-Nassr Club",
            end_anchor_text="Cristiano Ronaldo and John Duran up front",
            reason="Qwen coarse transcript anchors.",
        )
        second = replace(
            first,
            segment_id="lineup_002",
            team_name="Kawasaki Frontale",
            start_seconds=540.0,
            end_seconds=564.728,
            evidence_segment_ids=("chunk_009",),
            start_anchor_text="Has made six changes",
            end_anchor_text="gets his very first start",
        )
        result = LineupDetectionResult(
            model="qwen-plus",
            segments=(first, second),
            raw_response={},
            status="complete",
        )
        cuts = (
            454.5,
            459.033,
            474.8,
            475.467,
            484.833,
            489.6,
            493.8,
            498.333,
            503.333,
            505.633,
            533.067,
            534.733,
            544.633,
            548.3,
            551.533,
            554.667,
            557.2,
            563.0,
            565.433,
            568.633,
            571.6,
            574.2,
        )

        snapped = snap_lineup_result(result, cuts)

        self.assertEqual(snapped.segments[0].end_seconds, 505.633)
        self.assertEqual(snapped.segments[1].end_seconds, 565.433)
        self.assertEqual(
            snapped.raw_response["scene_refinement"]["segments"][0][
                "end_strategy"
            ],
            "near_audio_boundary",
        )


if __name__ == "__main__":
    unittest.main()
