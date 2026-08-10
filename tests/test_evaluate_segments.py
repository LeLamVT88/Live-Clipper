from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "lineup"))

from evaluate_segments import evaluate_video, load_predicted_segments


class SegmentInputTests(unittest.TestCase):
    def test_loads_detector_segment_contract(self) -> None:
        with TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "lineup_segments.csv"
            pd.DataFrame(
                [
                    {
                        "video": "match.mp4",
                        "start_seconds": 72.25,
                        "end_seconds": 98.5,
                        "confidence": 0.9,
                    }
                ]
            ).to_csv(csv_path, index=False)

            segments = load_predicted_segments(csv_path)

            self.assertEqual(segments.loc[0, "video"], "match.mp4")
            self.assertEqual(segments.loc[0, "start_seconds"], 72.25)
            self.assertEqual(segments.loc[0, "start"], "00:01:12.25")

    def test_rejects_reversed_segment(self) -> None:
        with TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "lineup_segments.csv"
            pd.DataFrame(
                [{"video": "match.mp4", "start": "00:02:00", "end": "00:01:00"}]
            ).to_csv(csv_path, index=False)

            with self.assertRaisesRegex(ValueError, "end after"):
                load_predicted_segments(csv_path)


class SegmentEvaluationTests(unittest.TestCase):
    def test_counts_video_with_no_predictions_as_missed(self) -> None:
        truth = [
            {"team": "Đội 1", "start_seconds": 60.0, "end_seconds": 90.0},
            {"team": "Đội 2", "start_seconds": 120.0, "end_seconds": 150.0},
        ]

        details, metrics = evaluate_video(
            video="match.mp4",
            ground_truth=truth,
            predictions=[],
            split_name="selected",
            iou_threshold=0.5,
        )

        self.assertEqual(len(details), 2)
        self.assertEqual(metrics["missed_segments"], 2)
        self.assertEqual(metrics["segment_recall"], 0.0)


if __name__ == "__main__":
    unittest.main()
