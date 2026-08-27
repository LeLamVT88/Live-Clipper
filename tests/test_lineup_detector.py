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
    LineupDetectionError,
    QwenLineupDetector,
    build_detection_prompt,
    write_lineup_csv,
)
from lineup.detect_from_transcript import build_parser as build_detection_parser
from lineup.schema import parse_detection_result
from transcript.schema import TranscriptSegment


class FakeQwenClient:
    def __init__(
        self,
        payload: dict[str, object],
        *,
        failures_before_success: int = 0,
    ) -> None:
        self.payload = payload
        self.failures_before_success = failures_before_success
        self.calls: list[dict[str, object]] = []

    def call(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if len(self.calls) <= self.failures_before_success:
            raise RuntimeError("503 UNAVAILABLE: model is overloaded")
        return SimpleNamespace(
            status_code=200,
            request_id="request-test",
            output={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                self.payload, ensure_ascii=False
                            )
                        }
                    }
                ]
            },
        )


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
        client = FakeQwenClient(
            {
                "lineup_segments": [
                    {
                        "start_seconds": 440,
                        "end_seconds": 468,
                        "confidence": 0.92,
                        "context": "lineup_presentation",
                        "team_name": "Team A",
                        "evidence_segment_ids": [
                            "chunk_001_seg_0001",
                            "chunk_001_seg_0002",
                        ],
                        "start_anchor_text": "Here is the starting eleven",
                        "end_anchor_text": (
                            "The goalkeeper is A, followed by B, C and D."
                        ),
                        "reason": "The commentator presents the starting eleven.",
                    }
                ]
            }
        )
        detector = QwenLineupDetector(model="qwen-test", client=client)

        result = detector.detect(
            self.transcript,
            video_name="match.mp4",
            window_start_seconds=0.0,
            window_end_seconds=600.0,
        )

        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.segments[0].segment_id, "lineup_001")
        self.assertEqual(result.segments[0].start_seconds, 440.0)
        call = client.calls[0]
        self.assertEqual(call["model"], "qwen-test")
        self.assertEqual(call["result_format"], "message")
        self.assertEqual(call["response_format"], {"type": "json_object"})
        self.assertIn("chunk_001_seg_0001", str(call["messages"]))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "lineup_segments.csv"
            write_lineup_csv(result, video_name="match.mp4", output_path=output)
            with output.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))

        self.assertEqual(rows[0]["video"], "match.mp4")
        self.assertEqual(rows[0]["team_name"], "Team A")
        self.assertEqual(rows[0]["detection_status"], "unconstrained")
        self.assertEqual(rows[0]["start_seconds"], "440.0")
        self.assertEqual(
            json.loads(rows[0]["evidence_segment_ids"]),
            ["chunk_001_seg_0001", "chunk_001_seg_0002"],
        )

    def test_rejects_evidence_not_present_in_transcript(self) -> None:
        client = FakeQwenClient(
            {
                "lineup_segments": [
                    {
                        "start_seconds": 440,
                        "end_seconds": 468,
                        "confidence": 0.8,
                        "context": "lineup_presentation",
                        "team_name": "Team A",
                        "evidence_segment_ids": ["invented_segment"],
                        "start_anchor_text": "Here is the starting eleven",
                        "end_anchor_text": "Here is the starting eleven",
                        "reason": "Lineup",
                    }
                ]
            }
        )
        detector = QwenLineupDetector(model="qwen-test", client=client)

        with self.assertRaisesRegex(LineupDetectionError, "invalid evidence"):
            detector.detect(
                self.transcript,
                video_name="match.mp4",
                window_start_seconds=0.0,
                window_end_seconds=600.0,
            )

    def test_rejects_lineup_without_exact_anchors(self) -> None:
        client = FakeQwenClient(
            {
                "lineup_segments": [
                    {
                        "start_seconds": 451,
                        "end_seconds": 455,
                        "confidence": 0.9,
                        "context": "lineup_presentation",
                        "team_name": "Team A",
                        "evidence_segment_ids": [
                            "chunk_001_seg_0001",
                            "chunk_001_seg_0002",
                        ],
                        "reason": "The chunk contains a lineup presentation.",
                    }
                ]
            }
        )
        detector = QwenLineupDetector(model="qwen-test", client=client)

        with self.assertRaisesRegex(LineupDetectionError, "both exact text anchors"):
            detector.detect(
                self.transcript,
                video_name="match.mp4",
                window_start_seconds=0.0,
                window_end_seconds=600.0,
            )

    def test_derives_coarse_timestamps_from_exact_text_anchors(self) -> None:
        text = "Intro. Here is the starting eleven. Referee details follow."
        transcript = (
            transcript_segment("chunk_000_seg_0001", 0.0, 60.0, text),
        )
        client = FakeQwenClient(
            {
                "lineup_segments": [
                    {
                        "start_seconds": 0.0,
                        "end_seconds": 60.0,
                        "confidence": 0.9,
                        "context": "lineup_presentation",
                        "team_name": "Team A",
                        "evidence_segment_ids": ["chunk_000_seg_0001"],
                        "reason": "A lineup appears before referee details.",
                        "start_anchor_text": "Here is the starting eleven",
                        "end_anchor_text": "Here is the starting eleven",
                    }
                ]
            }
        )

        result = QwenLineupDetector(model="qwen-test", client=client).detect(
            transcript,
            video_name="match.mp4",
            window_start_seconds=0.0,
            window_end_seconds=60.0,
        )

        expected_start = 60.0 * text.index("Here is") / len(text)
        expected_end = 60.0 * (
            text.index("Here is") + len("Here is the starting eleven")
        ) / len(text)
        self.assertAlmostEqual(result.segments[0].start_seconds, expected_start)
        self.assertAlmostEqual(result.segments[0].end_seconds, expected_end)

    def test_rejects_anchor_not_copied_from_cited_evidence(self) -> None:
        client = FakeQwenClient(
            {
                "lineup_segments": [
                    {
                        "start_seconds": 410,
                        "end_seconds": 430,
                        "confidence": 0.9,
                        "context": "lineup_presentation",
                        "team_name": "Team A",
                        "evidence_segment_ids": ["chunk_001_seg_0001"],
                        "start_anchor_text": "invented opening anchor",
                        "end_anchor_text": "invented closing anchor",
                        "reason": "The model timestamp is not supported.",
                    }
                ]
            }
        )
        detector = QwenLineupDetector(model="qwen-test", client=client)

        with self.assertRaisesRegex(
            LineupDetectionError,
            "not found in cited evidence",
        ):
            detector.detect(
                self.transcript,
                video_name="match.mp4",
                window_start_seconds=0.0,
                window_end_seconds=600.0,
            )

    def test_rejects_one_team_split_across_a_large_evidence_gap(self) -> None:
        transcript = (
            transcript_segment(
                "chunk_001_seg_0001",
                400.0,
                420.0,
                "Al Hilal starting lineup.",
            ),
            transcript_segment(
                "chunk_002_seg_0001",
                480.0,
                500.0,
                "Gwangju starting lineup.",
            ),
        )
        client = FakeQwenClient(
            {
                "lineup_segments": [
                    {
                        "start_seconds": 400,
                        "end_seconds": 500,
                        "confidence": 0.9,
                        "context": "lineup_presentation",
                        "team_name": "Al Hilal",
                        "evidence_segment_ids": [
                            "chunk_001_seg_0001",
                            "chunk_002_seg_0001",
                        ],
                        "start_anchor_text": "Al Hilal starting lineup",
                        "end_anchor_text": "Gwangju starting lineup",
                        "reason": "The model incorrectly merged two teams.",
                    }
                ]
            }
        )

        with self.assertRaisesRegex(LineupDetectionError, "distinct teams"):
            QwenLineupDetector(model="qwen-test", client=client).detect(
                transcript,
                video_name="match.mp4",
                window_start_seconds=0.0,
                window_end_seconds=600.0,
            )

    def test_empty_transcript_returns_empty_result_without_api_call(self) -> None:
        client = FakeQwenClient({"lineup_segments": []})
        detector = QwenLineupDetector(model="qwen-test", client=client)

        result = detector.detect(
            tuple(),
            video_name="match.mp4",
            window_start_seconds=0.0,
            window_end_seconds=600.0,
        )

        self.assertEqual(result.segments, tuple())
        self.assertEqual(client.calls, [])

    def test_cli_allows_unconstrained_lineup_count(self) -> None:
        args = build_detection_parser().parse_args(
            [
                "--transcript",
                "transcript.jsonl",
                "--expected-lineups",
                "auto",
            ]
        )

        self.assertIsNone(args.expected_lineups)

    def test_rejects_overlapping_lineups_for_distinct_teams(self) -> None:
        text = "Alpha lineup begins and ends. Beta lineup begins and ends."
        candidate = {
            "start_seconds": 0.0,
            "end_seconds": 60.0,
            "confidence": 0.9,
            "context": "lineup_presentation",
            "evidence_segment_ids": ["s1"],
            "start_anchor_text": "Alpha lineup begins",
            "end_anchor_text": "Alpha lineup begins",
            "reason": "Lineup presentation.",
        }
        payload = {
            "lineup_segments": [
                {**candidate, "team_name": "Alpha"},
                {**candidate, "team_name": "Beta"},
            ]
        }

        with self.assertRaisesRegex(LineupDetectionError, "must not overlap"):
            parse_detection_result(
                payload,
                model="qwen-test",
                known_evidence=["s1"],
                evidence_bounds={"s1": (0.0, 60.0)},
                evidence_texts={"s1": text},
                window_start=0.0,
                window_end=60.0,
            )

    def test_rejects_duplicate_team_names(self) -> None:
        text = "Alpha lineup begins and ends. Beta lineup begins and ends."
        common = {
            "start_seconds": 0.0,
            "end_seconds": 60.0,
            "confidence": 0.9,
            "context": "lineup_presentation",
            "team_name": "Same Team",
            "evidence_segment_ids": ["s1"],
            "reason": "Lineup presentation.",
        }
        payload = {
            "lineup_segments": [
                {
                    **common,
                    "start_anchor_text": "Alpha lineup begins",
                    "end_anchor_text": "Alpha lineup begins",
                },
                {
                    **common,
                    "start_anchor_text": "Beta lineup begins",
                    "end_anchor_text": "Beta lineup begins",
                },
            ]
        }

        with self.assertRaisesRegex(LineupDetectionError, "distinct teams"):
            parse_detection_result(
                payload,
                model="qwen-test",
                known_evidence=["s1"],
                evidence_bounds={"s1": (0.0, 60.0)},
                evidence_texts={"s1": text},
                window_start=0.0,
                window_end=60.0,
            )

    def test_prompt_contains_multilingual_transcript_without_labels(self) -> None:
        prompt = build_detection_prompt(
            self.transcript,
            video_name="match.mp4",
            window_start_seconds=0.0,
            window_end_seconds=600.0,
        )

        self.assertIn("starting eleven", prompt)
        self.assertIn("khoảng THÔ", prompt)
        self.assertIn("một block tốt nhất", prompt)
        self.assertIn("hai kết quả riêng", prompt)
        self.assertIn("Player walkout/ceremonial introduction", prompt)
        self.assertIn("Evidence của một kết quả phải liên tục", prompt)
        self.assertIn("lưu nguyên trạng", prompt)
        self.assertIn("start_anchor_text", prompt)
        self.assertIn('"required": ["lineup_segments"]', prompt)
        self.assertNotIn("{LINEUP_JSON_SCHEMA}", prompt)
        self.assertNotIn("07:22", prompt)

    def test_filters_live_play_tactical_analysis(self) -> None:
        client = FakeQwenClient(
            {
                "lineup_segments": [
                    {
                        "start_seconds": 440,
                        "end_seconds": 468,
                        "confidence": 0.95,
                        "context": "live_play_tactical_analysis",
                        "evidence_segment_ids": [
                            "chunk_001_seg_0001",
                            "chunk_001_seg_0002",
                        ],
                        "reason": (
                            "Play is under way and the commentator is "
                            "describing the current tactical shape."
                        ),
                    }
                ]
            }
        )
        detector = QwenLineupDetector(model="qwen-test", client=client)

        result = detector.detect(
            self.transcript,
            video_name="match.mp4",
            window_start_seconds=0.0,
            window_end_seconds=600.0,
        )

        self.assertEqual(result.segments, tuple())
        self.assertEqual(
            result.raw_response["lineup_segments"][0]["context"],
            "live_play_tactical_analysis",
        )

    def test_filters_ceremonial_walkout_but_not_last_outing(self) -> None:
        transcript = (
            transcript_segment(
                "chunk_001_seg_0001",
                300.0,
                360.0,
                "The goalkeeper will be next out. The defender is last out.",
            ),
            transcript_segment(
                "chunk_002_seg_0001",
                480.0,
                540.0,
                "Three changes from their last outing. Here is the lineup.",
            ),
        )
        client = FakeQwenClient(
            {
                "lineup_segments": [
                    {
                        "start_seconds": 300.0,
                        "end_seconds": 360.0,
                        "confidence": 0.9,
                        "context": "lineup_presentation",
                        "team_name": "Ceremonial Team",
                        "evidence_segment_ids": ["chunk_001_seg_0001"],
                        "start_anchor_text": "will be next out",
                        "end_anchor_text": "defender is last out",
                        "reason": "Ceremonial player names.",
                    },
                    {
                        "start_seconds": 480.0,
                        "end_seconds": 540.0,
                        "confidence": 0.9,
                        "context": "lineup_presentation",
                        "team_name": "Team B",
                        "evidence_segment_ids": ["chunk_002_seg_0001"],
                        "start_anchor_text": "Three changes from their last outing",
                        "end_anchor_text": "Here is the lineup",
                        "reason": "A real lineup after the last outing.",
                    },
                ]
            }
        )

        result = QwenLineupDetector(model="qwen-test", client=client).detect(
            transcript,
            video_name="match.mp4",
            window_start_seconds=0.0,
            window_end_seconds=600.0,
        )

        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.segments[0].start_seconds, 480.0)
        self.assertEqual(len(client.calls), 1)

    def test_keeps_real_lineup_after_walkout_in_the_same_chunk(self) -> None:
        text = (
            "The players will be next out. "
            "Here is the starting eleven for Team A."
        )
        transcript = (
            transcript_segment("chunk_001_seg_0001", 300.0, 360.0, text),
        )
        client = FakeQwenClient(
            {
                "lineup_segments": [
                    {
                        "team_name": "Team A",
                        "start_seconds": 300.0,
                        "end_seconds": 360.0,
                        "confidence": 0.9,
                        "context": "lineup_presentation",
                        "evidence_segment_ids": ["chunk_001_seg_0001"],
                        "start_anchor_text": "Here is the starting eleven",
                        "end_anchor_text": "starting eleven for Team A",
                        "reason": "A team-sheet follows the walkout.",
                    }
                ]
            }
        )

        result = QwenLineupDetector(model="qwen-test", client=client).detect(
            transcript,
            video_name="match.mp4",
            window_start_seconds=0.0,
            window_end_seconds=600.0,
        )

        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.segments[0].team_name, "Team A")

    def test_marks_incomplete_without_regenerating_to_force_count(self) -> None:
        client = FakeQwenClient(
            {
                "lineup_segments": [
                    {
                        "start_seconds": 440.0,
                        "end_seconds": 468.0,
                        "confidence": 0.9,
                        "context": "lineup_presentation",
                        "team_name": "Team A",
                        "evidence_segment_ids": [
                            "chunk_001_seg_0001",
                            "chunk_001_seg_0002",
                        ],
                        "start_anchor_text": "Here is the starting eleven",
                        "end_anchor_text": (
                            "The goalkeeper is A, followed by B, C and D."
                        ),
                        "reason": "Only one lineup was returned.",
                    }
                ]
            }
        )
        detector = QwenLineupDetector(
            model="qwen-test",
            client=client,
            max_attempts=2,
            retry_delay_seconds=0,
            expected_segment_count=2,
        )

        result = detector.detect(
            self.transcript,
            video_name="match.mp4",
            window_start_seconds=0.0,
            window_end_seconds=600.0,
        )

        self.assertEqual(result.status, "incomplete")
        self.assertEqual(len(result.segments), 1)
        self.assertEqual(len(client.calls), 1)

    def test_retries_transient_lineup_detection_failure(self) -> None:
        client = FakeQwenClient(
            {"lineup_segments": []},
            failures_before_success=1,
        )
        detector = QwenLineupDetector(
            model="qwen-test",
            client=client,
            max_attempts=2,
            retry_delay_seconds=0,
        )

        result = detector.detect(
            self.transcript,
            video_name="match.mp4",
            window_start_seconds=0.0,
            window_end_seconds=600.0,
        )

        self.assertEqual(result.segments, tuple())
        self.assertEqual(len(client.calls), 2)


if __name__ == "__main__":
    unittest.main()
