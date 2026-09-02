from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lineup.scene_detection import (
    SceneCut,
    _resolve_scene_min_length_frames,
    _soft_transition_cuts,
    snap_timestamp,
)
from lineup.schema import LineupDetectionError


class SceneDetectionTests(unittest.TestCase):
    def test_scene_minimum_duration_is_consistent_across_frame_rates(self) -> None:
        self.assertEqual(
            _resolve_scene_min_length_frames(
                frame_rate=25.0,
                min_scene_len_frames=None,
                min_scene_len_seconds=0.5,
            ),
            12,
        )
        self.assertEqual(
            _resolve_scene_min_length_frames(
                frame_rate=60.0,
                min_scene_len_frames=None,
                min_scene_len_seconds=0.5,
            ),
            30,
        )

    def test_explicit_frame_minimum_takes_precedence(self) -> None:
        self.assertEqual(
            _resolve_scene_min_length_frames(
                frame_rate=60.0,
                min_scene_len_frames=15,
                min_scene_len_seconds=0.5,
            ),
            15,
        )

    def test_invalid_scene_minimum_is_rejected(self) -> None:
        with self.assertRaises(LineupDetectionError):
            _resolve_scene_min_length_frames(
                frame_rate=0.0,
                min_scene_len_frames=None,
                min_scene_len_seconds=0.5,
            )

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
        self.assertTrue(all(isinstance(cut, SceneCut) for cut in cuts))

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
        nearby = snap_timestamp(
            568.2,
            (543.0, 568.834, 580.2),
            radius_seconds=3,
        )
        distant = snap_timestamp(
            575.0,
            (568.834, 580.2),
            radius_seconds=3,
        )

        self.assertTrue(nearby.applied)
        self.assertEqual(nearby.snapped_seconds, 568.834)
        self.assertFalse(distant.applied)
        self.assertEqual(distant.snapped_seconds, 575.0)

    def test_scene_snap_respects_boundary_direction(self) -> None:
        cuts = (539.767, 541.967, 568.833, 570.4)

        start = snap_timestamp(
            540,
            cuts,
            radius_seconds=4,
            preference="forward",
        )
        end = snap_timestamp(
            570,
            cuts,
            radius_seconds=4,
            preference="backward",
        )

        self.assertEqual(start.snapped_seconds, 541.967)
        self.assertEqual(end.snapped_seconds, 568.833)


if __name__ == "__main__":
    unittest.main()
