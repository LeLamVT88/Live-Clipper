from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from line_up.cli import run_full_pipeline
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
    def test_exports_detected_clips_and_simple_player_csv(self) -> None:
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
            with (
                patch(
                    "line_up.cli.detect_lineups",
                    return_value=detection([LineupInterval(10.0, 20.0, 0.95)]),
                ),
                patch("line_up.cli.export_video_clip", side_effect=export),
                patch("line_up.cli.LineupWorkflow", Workflow),
            ):
                status = run_full_pipeline(video, output_dir)

            with (output_dir / "players.csv").open(encoding="utf-8") as players:
                rows = list(csv.DictReader(players))
            self.assertEqual(status, 0)
            self.assertTrue((output_dir / "clips" / "match_lineup_1.mp4").is_file())
            self.assertEqual(
                rows,
                [{"lineup_clip": "match_lineup_1.mp4", "shirt_number": "9", "player_name": "KANE"}],
            )
            self.assertTrue((output_dir / "detection_result.json").is_file())

    def test_no_detection_writes_empty_players_csv_without_mapping(self) -> None:
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
                (output_dir / "players.csv").read_text(encoding="utf-8"),
                "lineup_clip,shirt_number,player_name\n",
            )
            payload = json.loads((output_dir / "detection_result.json").read_text())
            self.assertEqual(payload["lineups"], [])
            workflow.assert_not_called()


if __name__ == "__main__":
    unittest.main()
