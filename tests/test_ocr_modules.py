from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mapping.engine import detections_from_result
from mapping.resolver.pipeline import resolve_all_lineups


class FakeOCRResult:
    def __init__(self, payload: dict[str, object]) -> None:
        self.json = {"res": payload}


class PaddleRunnerTests(unittest.TestCase):
    def test_converts_paddle_result_to_normalized_detection(self) -> None:
        frame = {
            "video": "match.mp4",
            "segment_index": 1,
            "segment_label": "lineup_01",
            "segment_start_seconds": 10.0,
            "segment_end_seconds": 20.0,
            "frame_index": 1,
            "frame_path": "frame.jpg",
            "timestamp": "00:00:10",
            "timestamp_seconds": 10.0,
            "relative_seconds": 0.0,
            "frame_width": 1000,
            "frame_height": 500,
        }
        result = FakeOCRResult(
            {
                "rec_texts": ["9", "LOW SCORE"],
                "rec_scores": [0.99, 0.50],
                "rec_boxes": [
                    [100, 100, 140, 140],
                    [200, 200, 300, 240],
                ],
            }
        )

        detections = detections_from_result(
            frame,
            result,
            min_score=0.80,
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(
            detections[0]["text_type"],
            "shirt_number_candidate",
        )
        self.assertAlmostEqual(
            float(detections[0]["center_x_norm"]),
            0.12,
        )
        self.assertAlmostEqual(
            float(detections[0]["center_y_norm"]),
            0.24,
        )


class ResolverPipelineTests(unittest.TestCase):
    def test_resolves_repeated_complete_table(self) -> None:
        names = [
            "ALPHA",
            "BRAVO",
            "CHARLIE",
            "DELTA",
            "ECHO",
            "FOXTROT",
            "GOLF",
            "HOTEL",
            "INDIA",
            "JULIET",
            "KILO",
        ]
        rows: list[dict[str, object]] = []
        for frame_index in (1, 2):
            for shirt_number, name in enumerate(names, start=1):
                rows.append(
                    {
                        "video": "match.mp4",
                        "segment_index": 1,
                        "segment_start_seconds": 10.0,
                        "segment_end_seconds": 20.0,
                        "frame_index": frame_index,
                        "timestamp_seconds": (10.0 + frame_index * 0.5),
                        "text": f"{shirt_number} {name}",
                        "text_type": "text",
                        "score": 0.99,
                        "center_x_norm": 0.20,
                        "center_y_norm": (0.15 + shirt_number * 0.05),
                    }
                )

        records, diagnostics = resolve_all_lineups(
            pd.DataFrame(rows),
            expected_players=11,
            min_number_count=8,
            max_gap_seconds=20.0,
            signature_threshold=0.45,
            enable_local_ocr=False,
        )

        self.assertEqual(len(records), 11)
        self.assertEqual(
            diagnostics[0]["status"],
            "resolved",
        )
        self.assertEqual(
            diagnostics[0]["resolution_method"],
            "table",
        )


if __name__ == "__main__":
    unittest.main()
