from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from line_up.cli import PROJECT_ROOT, build_parser, default_output_dir, run_full_pipeline
from line_up.config import PipelineConfig
from line_up.pipeline import detect_lineups
from line_up.schema import DetectionResult, LineupInterval


def detection(lineups: list[LineupInterval]) -> DetectionResult:
    return DetectionResult(
        video_path=Path("match.mp4"),
        video_duration_scanned=120.0,
        scenes=[(0.0, 120.0)],
        sampled_frames=[],
        lineups=lineups,
        processing_time_seconds=1.25,
    )


class FullPipelineTests(unittest.TestCase):
    def test_default_output_preserves_league_and_match_from_data_path(self) -> None:
        video = PROJECT_ROOT / "data" / "premier" / "premier_match_01.mp4"

        self.assertEqual(
            default_output_dir(video),
            PROJECT_ROOT / "outputs" / "premier" / "premier_match_01",
        )

    def test_defaults_to_scanning_the_complete_video(self) -> None:
        args = build_parser().parse_args(["match.mp4"])

        self.assertIsNone(args.max_scan)
        self.assertIsNone(PipelineConfig().max_scan_seconds)

    def test_detector_uses_complete_video_duration_when_uncapped(self) -> None:
        class Capture:
            def get(self, prop: int) -> float:
                return 25.0 if prop == 5 else 18_000.0

            def release(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "match.mp4"
            video.write_bytes(b"video")
            with (
                patch("line_up.pipeline.cv2.VideoCapture", return_value=Capture()),
                patch("line_up.pipeline.detect_scenes", return_value=[]) as scenes,
                patch("line_up.pipeline.get_scene_sample_timestamps", return_value=[]),
                patch("line_up.pipeline.get_ocr_model"),
                patch("line_up.pipeline.extract_frames_at_timestamps", return_value=([], [])),
                patch("line_up.pipeline.run_ocr_batch", return_value=[]),
            ):
                result = detect_lineups(video, PipelineConfig(use_multimodal=False))

        self.assertEqual(result.video_duration_scanned, 720.0)
        self.assertEqual(scenes.call_args.kwargs["max_duration"], 720.0)

    def test_exports_detected_clips_and_simple_player_json(self) -> None:
        class Workflow:
            def __init__(self, config):
                self.config = config

            def run(self) -> int:
                self.config.resolved_output_csv.parent.mkdir(parents=True, exist_ok=True)
                with self.config.resolved_output_csv.open("w", newline="", encoding="utf-8") as output:
                    writer = csv.DictWriter(
                        output,
                        fieldnames=("video", "shirt_number", "player_name"),
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "video": "outputs/match/clips/match_lineup_1.mp4",
                            "shirt_number": 9,
                            "player_name": "KANE",
                        }
                    )
                return 0

        def export(_video, _start, _end, output_path) -> bool:
            output_path.write_bytes(b"clip")
            return True

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "match.mp4"
            video.write_bytes(b"video")
            output_dir = root / "result"
            clips_dir = output_dir / "clips"
            clips_dir.mkdir(parents=True)
            stale_clip = clips_dir / "match_lineup_2.mp4"
            stale_clip.write_bytes(b"stale false positive")
            unrelated_clip = clips_dir / "other_lineup_1.mp4"
            unrelated_clip.write_bytes(b"unrelated")
            with (
                patch(
                    "line_up.cli.detect_lineups",
                    return_value=detection([LineupInterval(10.0, 20.0, 0.95)]),
                ),
                patch("line_up.cli.export_video_clip", side_effect=export),
                patch("line_up.cli.LineupWorkflow", Workflow),
            ):
                status = run_full_pipeline(video, output_dir)

            rows = json.loads((output_dir / "players.json").read_text(encoding="utf-8"))
            self.assertEqual(status, 0)
            self.assertTrue((output_dir / "clips" / "match_lineup_1.mp4").is_file())
            self.assertFalse(stale_clip.exists())
            self.assertTrue(unrelated_clip.is_file())
            self.assertEqual(
                rows,
                [{
                    "lineup_clip": "match_lineup_1.mp4", "lineup_index": 1,
                    "shirt_number": 9, "player_name": "KANE",
                }],
            )
            self.assertTrue((output_dir / "detection_result.json").is_file())
            self.assertFalse((output_dir / "players.csv").exists())
            self.assertFalse((output_dir / "resolved_lineups.csv").exists())

    def test_no_detection_writes_empty_players_json_without_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "match.mp4"
            video.write_bytes(b"video")
            output_dir = root / "result"
            with (
                patch("line_up.cli.detect_lineups", return_value=detection([])),
                patch("line_up.cli.LineupWorkflow") as workflow,
            ):
                status = run_full_pipeline(video, output_dir)

            self.assertEqual(status, 2)
            self.assertEqual(
                json.loads((output_dir / "players.json").read_text(encoding="utf-8")), []
            )
            payload = json.loads((output_dir / "detection_result.json").read_text())
            self.assertEqual(payload["lineups"], [])
            workflow.assert_not_called()

    def test_keep_diagnostics_preserves_mapping_csv(self) -> None:
        class Workflow:
            def __init__(self, config):
                self.config = config

            def run(self) -> int:
                self.config.resolved_output_csv.parent.mkdir(parents=True, exist_ok=True)
                with self.config.resolved_output_csv.open("w", newline="", encoding="utf-8") as output:
                    writer = csv.DictWriter(
                        output, fieldnames=("video", "shirt_number", "player_name")
                    )
                    writer.writeheader()
                    writer.writerow(
                        {"video": "match_lineup_1.mp4", "shirt_number": 1, "player_name": "KEEPER"}
                    )
                return 0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "match.mp4"
            video.write_bytes(b"video")
            output_dir = root / "result"
            with (
                patch(
                    "line_up.cli.detect_lineups",
                    return_value=detection([LineupInterval(10.0, 20.0, 0.95)]),
                ),
                patch("line_up.cli.export_video_clip", return_value=True),
                patch("line_up.cli.LineupWorkflow", Workflow),
            ):
                status = run_full_pipeline(video, output_dir, keep_diagnostics=True)

            self.assertEqual(status, 0)
            self.assertTrue((output_dir / "resolved_lineups.csv").is_file())


if __name__ == "__main__":
    unittest.main()
