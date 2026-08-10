from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lineup.detector import (
    GeminiLineupDetector,
    LineupDetectionError,
    build_detection_prompt,
    write_lineup_csv,
)
from transcript.schema import TranscriptSegment


class FakeModels:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(text=json.dumps(self.payload, ensure_ascii=False))


class FakeGeminiClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.models = FakeModels(payload)


def transcript_segment(
    segment_id: str,
    start_seconds: float,
    end_seconds: float,
    text: str,
) -> TranscriptSegment:
    return TranscriptSegment(
        segment_id=segment_id,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        text=text,
        source_chunk="chunk_001",
        chunk_start_seconds=300.0,
    )


class LineupDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transcript = (
            transcript_segment(
                "chunk_001_seg_0001",
                440.0,
                448.0,
                "Here is the starting eleven.",
            ),
            transcript_segment(
                "chunk_001_seg_0002",
                448.0,
                468.0,
                "The goalkeeper is A, followed by B, C and D.",
            ),
        )

    def test_detects_orders_and_persists_structured_lineup_segments(self) -> None:
        client = FakeGeminiClient(
            {
                "lineup_segments": [
                    {
                        "start_seconds": 440,
                        "end_seconds": 468,
                        "confidence": 0.92,
                        "evidence_segment_ids": [
                            "chunk_001_seg_0001",
                            "chunk_001_seg_0002",
                        ],
                        "reason": "The commentator presents the starting eleven.",
                    }
                ]
            }
        )
        detector = GeminiLineupDetector(model="gemini-test", client=client)

        result = detector.detect(
            self.transcript,
            video_name="match.mp4",
            window_start_seconds=0.0,
            window_end_seconds=600.0,
        )

        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.segments[0].segment_id, "lineup_001")
        self.assertEqual(result.segments[0].start_seconds, 440.0)
        call = client.models.calls[0]
        self.assertEqual(call["model"], "gemini-test")
        self.assertNotIn("ground_truth", str(call["contents"]).lower())
        self.assertIn("chunk_001_seg_0001", str(call["contents"]))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "lineup_segments.csv"
            write_lineup_csv(result, video_name="match.mp4", output_path=output)
            with output.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(rows[0]["video"], "match.mp4")
        self.assertEqual(rows[0]["start_seconds"], "440.0")
        self.assertEqual(
            json.loads(rows[0]["evidence_segment_ids"]),
            ["chunk_001_seg_0001", "chunk_001_seg_0002"],
        )

    def test_rejects_evidence_not_present_in_transcript(self) -> None:
        client = FakeGeminiClient(
            {
                "lineup_segments": [
                    {
                        "start_seconds": 440,
                        "end_seconds": 468,
                        "confidence": 0.8,
                        "evidence_segment_ids": ["invented_segment"],
                        "reason": "Lineup",
                    }
                ]
            }
        )
        detector = GeminiLineupDetector(model="gemini-test", client=client)

        with self.assertRaisesRegex(LineupDetectionError, "unknown evidence"):
            detector.detect(
                self.transcript,
                video_name="match.mp4",
                window_start_seconds=0.0,
                window_end_seconds=600.0,
            )

    def test_empty_transcript_returns_empty_result_without_api_call(self) -> None:
        client = FakeGeminiClient({"lineup_segments": []})
        detector = GeminiLineupDetector(model="gemini-test", client=client)

        result = detector.detect(
            tuple(),
            video_name="match.mp4",
            window_start_seconds=0.0,
            window_end_seconds=600.0,
        )

        self.assertEqual(result.segments, tuple())
        self.assertEqual(client.models.calls, [])

    def test_prompt_contains_multilingual_transcript_without_labels(self) -> None:
        prompt = build_detection_prompt(
            self.transcript,
            video_name="match.mp4",
            window_start_seconds=0.0,
            window_end_seconds=600.0,
        )

        self.assertIn("starting eleven", prompt)
        self.assertIn("any language", prompt)
        self.assertNotIn("07:22", prompt)


if __name__ == "__main__":
    unittest.main()
