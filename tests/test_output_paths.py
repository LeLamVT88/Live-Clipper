from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lineup.utils import (
    default_lineup_clip_dir,
    default_lineup_prediction_dir,
    default_run_dir,
)


class OutputPathTests(unittest.TestCase):
    def test_run_dir_mirrors_the_raw_data_hierarchy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_data = root / "data" / "raw_data"
            output_root = root / "outputs"
            video = raw_data / "ACLE" / "ACLE_01.mp4"

            run_dir = default_run_dir(
                video,
                raw_data_root=raw_data,
                output_root=output_root,
            )

            self.assertEqual(
                run_dir,
                (output_root / "ACLE" / "ACLE_01").resolve(),
            )

    def test_run_dir_preserves_nested_league_folders(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_data = root / "data" / "raw_data"
            output_root = root / "outputs"
            video = raw_data / "international" / "UCL" / "UCL_01.mp4"

            run_dir = default_run_dir(
                video,
                raw_data_root=raw_data,
                output_root=output_root,
            )

            self.assertEqual(
                run_dir,
                (
                    output_root / "international" / "UCL" / "UCL_01"
                ).resolve(),
            )

    def test_video_outside_raw_data_uses_the_legacy_flat_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "outputs"

            run_dir = default_run_dir(
                root / "uploads" / "Friendly Match.mp4",
                raw_data_root=root / "data" / "raw_data",
                output_root=output_root,
            )

            self.assertEqual(
                run_dir,
                (output_root / "Friendly_Match").resolve(),
            )

    def test_detection_and_export_stay_inside_the_same_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "outputs" / "ACLE" / "ACLE_01"
            transcript = (
                run_dir
                / "predictions"
                / "transcript"
                / "transcript.jsonl"
            )
            lineup_csv = (
                default_lineup_prediction_dir(transcript)
                / "lineup_segments.csv"
            )

            self.assertEqual(
                lineup_csv,
                (
                    run_dir
                    / "predictions"
                    / "lineup"
                    / "lineup_segments.csv"
                ).resolve(),
            )
            self.assertEqual(
                default_lineup_clip_dir(lineup_csv),
                (run_dir / "clips" / "lineup").resolve(),
            )


if __name__ == "__main__":
    unittest.main()
