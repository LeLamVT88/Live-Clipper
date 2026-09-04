from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lineup.detect_from_video import build_parser
from lineup.ocr_worker import Detection, Recognition, Unit, _events
from lineup.video_detection import detect_lineups_from_video


class VideoLineupDetectionTests(unittest.TestCase):
    def test_cli_requires_only_a_video_and_accepts_full_scan(self) -> None:
        args = build_parser().parse_args(
            ["--source-video", "match.mp4", "--scan-end", "full"]
        )

        self.assertEqual(args.source_video, Path("match.mp4"))
        self.assertIsNone(args.scan_end)
        self.assertEqual(args.expected_lineups, 2)

    def test_two_pass_detection_uses_dense_bounds_and_adds_padding(self) -> None:
        calls: list[tuple[str, float, int]] = []

        def fake_scene_detector(
            _video: Path,
            *,
            start_seconds: float,
            end_seconds: float,
            **_kwargs: object,
        ) -> tuple[float, ...]:
            self.assertEqual((start_seconds, end_seconds), (0.0, 180.0))
            return (90.0, 140.0)

        def fake_ocr_runner(
            _video: Path,
            tasks: object,
            *,
            max_recognition_frames: int,
            **_kwargs: object,
        ) -> dict[str, object]:
            task = tuple(tasks)[0]
            unit_max = max(
                unit.end_seconds - unit.start_seconds for unit in task.units
            )
            calls.append((task.mode, unit_max, max_recognition_frames))
            if task.mode == "global":
                event = {
                    "start_seconds": 100.0,
                    "end_seconds": 130.0,
                    "confidence": 0.82,
                    "sample_seconds": [106.5, 124.5],
                    "texts": ["TEAM A", "4-3-3", "1 KIM"],
                }
            else:
                event = {
                    "start_seconds": 102.0,
                    "end_seconds": 128.0,
                    "confidence": 0.91,
                    "sample_seconds": [102.25, 127.75],
                    "texts": ["TEAM A", "10 PARK"],
                }
            return {
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "mode": task.mode,
                        "events": [event],
                    }
                ]
            }

        result = detect_lineups_from_video(
            Path("unused.mp4"),
            scan_start_seconds=0.0,
            scan_end_seconds=180.0,
            expected_count=1,
            scene_threshold=27.0,
            scene_min_length_frames=None,
            scene_min_length_seconds=0.5,
            scene_snap_radius_seconds=4.0,
            scene_detector=fake_scene_detector,
            ocr_runner=fake_ocr_runner,
        )

        self.assertEqual([call[0] for call in calls], ["global", "dense"])
        self.assertLessEqual(calls[0][1], 3.0)
        self.assertLessEqual(calls[1][1], 0.5)
        self.assertGreaterEqual(calls[0][2], 60)
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.segments[0].team_name, "TEAM A")
        self.assertEqual(result.segments[0].start_seconds, 98.0)
        self.assertEqual(result.segments[0].end_seconds, 134.0)
        self.assertEqual(result.segments[0].evidence_start_seconds, 102.0)
        self.assertEqual(result.segments[0].evidence_end_seconds, 128.0)

    def test_dense_miss_keeps_the_coarse_candidate_for_recall(self) -> None:
        def fake_scene_detector(*_args: object, **_kwargs: object) -> tuple[float, ...]:
            return ()

        def fake_ocr_runner(
            _video: Path, tasks: object, **_kwargs: object
        ) -> dict[str, object]:
            task = tuple(tasks)[0]
            events = (
                [
                    {
                        "start_seconds": 20.0,
                        "end_seconds": 40.0,
                        "confidence": 0.8,
                        "texts": ["TEAM B", "STARTING XI"],
                    }
                ]
                if task.mode == "global"
                else []
            )
            return {"tasks": [{"task_id": task.task_id, "events": events}]}

        result = detect_lineups_from_video(
            Path("unused.mp4"),
            scan_start_seconds=0.0,
            scan_end_seconds=100.0,
            expected_count=1,
            scene_threshold=27.0,
            scene_min_length_frames=None,
            scene_min_length_seconds=0.5,
            scene_snap_radius_seconds=4.0,
            scene_detector=fake_scene_detector,
            ocr_runner=fake_ocr_runner,
        )

        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.segments[0].start_seconds, 16.0)
        self.assertEqual(result.segments[0].end_seconds, 46.0)
        self.assertIn(
            "dense_ocr_did_not_reconfirm_all_candidates",
            result.raw_response["_review_reasons"],
        )

    def test_player_by_player_sequence_is_promoted_and_expanded_both_ways(self) -> None:
        units = tuple(
            Unit(
                index,
                index * 3.0,
                (index + 1) * 3.0,
                index * 3.0 + 1.5,
                0 if index == 0 else 2 if index == 5 else 1,
            )
            for index in range(6)
        )
        graphic = Detection(
            box_count=5,
            area_ratio=0.03,
            x_span=0.5,
            y_span=0.5,
            layout_cells=(9, 17, 25),
            appearance_histogram=(1.0, 0.0),
        )
        detections = {
            0: Detection(1, 0.002, appearance_histogram=(0.0, 1.0)),
            1: graphic,
            2: graphic,
            3: graphic,
            4: graphic,
            5: Detection(1, 0.002, appearance_histogram=(0.0, 1.0)),
        }
        recognitions = {
            1: Recognition(False, 0.31, ("1 KIM", "GOALKEEPER")),
            2: Recognition(False, 0.34, ("7 LEE", "MIDFIELDER")),
            3: Recognition(False, 0.32, ("10 PARK", "FORWARD")),
        }

        events = _events(
            "global_fallback",
            units,
            detections,
            recognitions,
            center_seconds=9.0,
            max_events=2,
            prefer_center=False,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["start_seconds"], 3.0)
        self.assertEqual(events[0]["end_seconds"], 15.0)


if __name__ == "__main__":
    unittest.main()
