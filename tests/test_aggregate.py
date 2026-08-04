from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "lineup"))

from aggregate import force_segment_count


class ForcedSegmentCountTests(unittest.TestCase):
    def test_splits_one_segment_at_a_local_score_valley(self) -> None:
        timestamps = list(range(140, 196, 2))
        scores = [0.9] * len(timestamps)
        for timestamp, score in ((168, 0.8), (170, 0.2), (172, 0.3), (174, 0.9)):
            scores[timestamps.index(timestamp)] = score
        predictions = pd.DataFrame(
            {
                "video": "match.mp4",
                "timestamp_seconds": timestamps,
                "score": scores,
            }
        )
        segments = [
            {
                "video": "match.mp4",
                "start": "00:02:20",
                "end": "00:03:14",
                "start_seconds": 140.0,
                "end_seconds": 194.0,
            }
        ]

        result = force_segment_count(
            segments,
            predictions,
            expected_count=2,
            min_duration_seconds=8.0,
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["end_seconds"], 170.0)
        self.assertEqual(result[1]["start_seconds"], 170.0)


if __name__ == "__main__":
    unittest.main()
