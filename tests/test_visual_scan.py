from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lineup.ocr_worker import (
    Detection,
    Recognition,
    Unit,
    _events,
    _middle_out,
    _player_slide_seed_indices,
    _recognition_score,
    _shortlist,
)
from lineup.visual_scan import VisualEvent, build_visual_units, snap_visual_event


class VisualScanTests(unittest.TestCase):
    def test_middle_out_order_is_available_for_local_diagnostics(self) -> None:
        units = (
            Unit(0, 0.0, 3.0, 1.5),
            Unit(1, 3.0, 6.0, 4.5),
            Unit(2, 6.0, 9.0, 7.5),
        )
        self.assertEqual(_middle_out(units, 4.5), (1, 0, 2))

    def test_ocr_gate_rejects_scoreboard_and_accepts_roster(self) -> None:
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

    def test_long_scenes_have_no_representative_gap_over_three_seconds(self) -> None:
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

    def test_event_expands_one_supported_unit_to_the_right(self) -> None:
        units = tuple(
            Unit(index, index * 3.0, (index + 1) * 3.0, index * 3.0 + 1.5, 0)
            for index in range(6)
        )
        graphic = Detection(
            box_count=9,
            area_ratio=0.03,
            x_span=0.6,
            y_span=0.7,
            layout_cells=(0, 8, 16, 24),
        )
        recognitions = {
            2: Recognition(True, 0.8, ("4-4-2", "1 KIM", "2 CHO")),
            3: Recognition(True, 0.8, ("SUBSTITUTES", "9 LEE", "10 PARK")),
        }
        events = _events(
            "global_scan",
            units,
            {index: graphic for index in range(6)},
            recognitions,
            center_seconds=9.0,
            max_events=1,
            prefer_center=False,
        )
        self.assertEqual(events[0]["start_seconds"], 0.0)
        self.assertEqual(events[0]["end_seconds"], 15.0)

    def test_scene_snap_only_expands_outward(self) -> None:
        event = VisualEvent(133.152, 164.12, 0.9, "lineup_001")
        snapped = snap_visual_event(
            event,
            (136.0, 138.0, 164.12),
            radius_seconds=4.0,
        )
        self.assertEqual(snapped.start_seconds, 133.152)
        self.assertEqual(snapped.end_seconds, 164.12)

    def test_officials_and_player_stats_are_hard_event_barriers(self) -> None:
        units = tuple(
            Unit(index, index * 3.0, (index + 1) * 3.0, index * 3.0 + 1.5, index)
            for index in range(5)
        )
        layout = Detection(
            box_count=12,
            area_ratio=0.04,
            x_span=0.7,
            y_span=0.7,
            layout_cells=(8, 16, 24, 32),
        )
        recognitions = {
            1: Recognition(True, 0.8, ("BELGIUM", "4-3-3", "1 COURTOIS")),
            2: Recognition(
                False,
                0.35,
                ("MATCH OFFICIALS", "1 REFEREE", "2 ASSISTANT"),
            ),
            3: Recognition(
                False,
                0.32,
                ("9 LUKAKU", "4 goals from 5 attempts"),
            ),
        }
        events = _events(
            "global_scan",
            units,
            {index: layout for index in range(5)},
            recognitions,
            center_seconds=5.0,
            max_events=2,
            prefer_center=False,
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["end_seconds"], 6.0)

    def test_location_transition_is_not_a_player_slide_sequence(self) -> None:
        units = tuple(
            Unit(index, index * 2.0, (index + 1) * 2.0, index * 2.0 + 1.0, index)
            for index in range(5)
        )
        layout = Detection(
            box_count=10,
            area_ratio=0.04,
            x_span=0.8,
            y_span=0.7,
            layout_cells=(8, 16, 24, 32),
        )
        recognitions = {
            0: Recognition(True, 0.8, ("BELGIUM", "4-3-3", "1 COURTOIS")),
            1: Recognition(False, 0.3, ("FIFA", "10", "ROSTOV-ON-DON")),
            2: Recognition(
                False, 0.3, ("MATCH OFFICIALS", "1 REFEREE")
            ),
            3: Recognition(
                False, 0.3, ("MATCH OFFICIALS", "2 ASSISTANT")
            ),
            4: Recognition(
                False, 0.3, ("MATCH OFFICIALS", "3 ASSISTANT")
            ),
        }
        promoted = _player_slide_seed_indices(
            units,
            {index: layout for index in range(5)},
            recognitions,
            {2, 3, 4},
        )
        events = _events(
            "global_scan",
            units,
            {index: layout for index in range(5)},
            recognitions,
            center_seconds=1.0,
            max_events=2,
            prefer_center=False,
        )
        self.assertNotIn(1, promoted)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["end_seconds"], 4.0)


if __name__ == "__main__":
    unittest.main()
