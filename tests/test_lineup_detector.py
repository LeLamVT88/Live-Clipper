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
    DEFAULT_MAX_COARSE_LINEUP_DURATION_SECONDS,
    DEFAULT_TEXT_BASE_URL,
    DEFAULT_TEXT_MODEL,
    LineupDetectionError,
    OpenAICompatibleLineupClient,
    QwenLineupDetector,
    _response_payload,
    build_detection_prompt,
    write_lineup_csv,
)
from lineup.detect_from_transcript import (
    _request_summary,
    build_parser as build_detection_parser,
)
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


class SequenceQwenClient(FakeQwenClient):
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        if not payloads:
            raise ValueError("payloads cannot be empty")
        super().__init__(payloads[-1])
        self.payloads = payloads

    def call(self, **kwargs: object) -> object:
        payload_index = min(len(self.calls), len(self.payloads) - 1)
        self.payload = self.payloads[payload_index]
        return super().call(**kwargs)


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
        self.assertFalse(call["enable_thinking"])
        self.assertIsInstance(call["messages"][0]["content"], str)
        self.assertEqual(call["temperature"], 0.0)
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

    def test_openai_compatible_client_sends_one_text_only_request(self) -> None:
        captured: dict[str, object] = {}

        class FakeHTTPResponse:
            status = 200

            def __enter__(self) -> "FakeHTTPResponse":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            @staticmethod
            def read() -> bytes:
                return json.dumps(
                    {
                        "id": "local-request",
                        "choices": [
                            {
                                "message": {
                                    "content": '{"lineup_segments": []}'
                                }
                            }
                        ],
                    }
                ).encode("utf-8")

        def fake_open(request: object, *, timeout: float) -> FakeHTTPResponse:
            captured["url"] = request.full_url  # type: ignore[attr-defined]
            captured["body"] = json.loads(  # type: ignore[attr-defined]
                request.data.decode("utf-8")
            )
            captured["timeout"] = timeout
            return FakeHTTPResponse()

        client = OpenAICompatibleLineupClient(
            base_url="http://localhost:8000/v1/",
            opener=fake_open,
        )
        response = client.call(
            model=DEFAULT_TEXT_MODEL,
            messages=[{"role": "user", "content": "transcript text"}],
            temperature=0.0,
            max_tokens=2048,
            result_format="message",
            response_format={"type": "json_object"},
            enable_thinking=False,
        )

        self.assertEqual(captured["url"], "http://localhost:8000/v1/chat/completions")
        self.assertEqual(captured["body"]["messages"][0]["content"], "transcript text")
        self.assertNotIn("result_format", captured["body"])
        self.assertEqual(
            captured["body"]["response_format"], {"type": "json_object"}
        )
        self.assertFalse(captured["body"]["enable_thinking"])
        self.assertEqual(response["status_code"], 200)

    def test_accepts_json_wrapped_in_a_single_markdown_fence(self) -> None:
        response = SimpleNamespace(
            status_code=200,
            output={
                "choices": [
                    {
                        "message": {
                            "content": "```json\n{\"lineup_segments\": []}\n```"
                        }
                    }
                ]
            },
        )

        self.assertEqual(_response_payload(response), {"lineup_segments": []})

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

    def test_matches_anchor_words_across_punctuation_differences(self) -> None:
        text = "Starting eleven: goalkeeper A, defenders B-C and D."
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
                        "reason": "Direct roster presentation.",
                        "start_anchor_text": (
                            "Starting eleven goalkeeper A"
                        ),
                        "end_anchor_text": "defenders B C and D",
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

        self.assertEqual(len(result.segments), 1)
        self.assertLess(result.segments[0].end_seconds, 60.0)

    def test_recovers_uniquely_missing_evidence_for_an_exact_anchor(self) -> None:
        transcript = (
            transcript_segment(
                "chunk_008_seg_0001",
                480.0,
                510.0,
                "Al Nassr make two changes to their starting lineup.",
            ),
            transcript_segment(
                "chunk_009_seg_0001",
                510.0,
                540.0,
                "Sadio Mane and John Duran return to the front line.",
            ),
        )
        client = FakeQwenClient(
            {
                "lineup_segments": [
                    {
                        "start_seconds": 480.0,
                        "end_seconds": 540.0,
                        "confidence": 0.9,
                        "context": "lineup_presentation",
                        "team_name": "Al Nassr",
                        # Qwen omitted the segment containing the end anchor.
                        "evidence_segment_ids": ["chunk_008_seg_0001"],
                        "start_anchor_text": (
                            "Al Nassr make two changes to their starting lineup"
                        ),
                        "end_anchor_text": (
                            "Sadio Mane and John Duran return to the front line"
                        ),
                        "reason": "A continuous direct roster presentation.",
                    }
                ]
            }
        )

        result = QwenLineupDetector(model="qwen-test", client=client).detect(
            transcript,
            video_name="match.mp4",
            window_start_seconds=480.0,
            window_end_seconds=540.0,
        )

        self.assertEqual(
            result.segments[0].evidence_segment_ids,
            ("chunk_008_seg_0001", "chunk_009_seg_0001"),
        )
        self.assertGreater(result.segments[0].end_seconds, 510.0)

    def test_regenerates_a_coarse_lineup_longer_than_the_limit(self) -> None:
        text = (
            "Lineup begins. "
            + "background words " * 80
            + "Compact direct roster."
        )
        transcript = (
            transcript_segment("chunk_000_seg_0001", 0.0, 120.0, text),
        )
        common = {
            "start_seconds": 0.0,
            "end_seconds": 120.0,
            "confidence": 0.9,
            "context": "lineup_presentation",
            "team_name": "Team A",
            "evidence_segment_ids": ["chunk_000_seg_0001"],
            "reason": "A direct roster presentation.",
        }
        client = SequenceQwenClient(
            [
                {
                    "lineup_segments": [
                        {
                            **common,
                            "start_anchor_text": "Lineup begins",
                            "end_anchor_text": "Compact direct roster",
                        }
                    ]
                },
                {
                    "lineup_segments": [
                        {
                            **common,
                            "start_anchor_text": "Compact direct roster",
                            "end_anchor_text": "Compact direct roster",
                        }
                    ]
                },
            ]
        )
        detector = QwenLineupDetector(
            model="qwen-test",
            client=client,
            max_attempts=2,
            retry_delay_seconds=0,
        )

        result = detector.detect(
            transcript,
            video_name="match.mp4",
            window_start_seconds=0.0,
            window_end_seconds=120.0,
        )

        self.assertEqual(len(client.calls), 2)
        repair_prompt = str(client.calls[1]["messages"])
        self.assertIn("LƯỢT SỬA BẮT BUỘC", repair_prompt)
        self.assertIn("coarse duration", repair_prompt)
        self.assertLessEqual(
            result.segments[0].end_seconds
            - result.segments[0].start_seconds,
            DEFAULT_MAX_COARSE_LINEUP_DURATION_SECONDS,
        )
        self.assertEqual(
            result.raw_response["_validation_attempts"][0]["status"],
            "invalid_payload",
        )

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

        with self.assertRaisesRegex(LineupDetectionError, "coarse duration"):
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

    def test_cli_defaults_to_qwen_plus_without_short_asr_options(self) -> None:
        parser = build_detection_parser()
        args = parser.parse_args(["--transcript", "transcript.jsonl"])

        self.assertEqual(DEFAULT_TEXT_MODEL, "qwen-plus")
        self.assertEqual(
            DEFAULT_TEXT_BASE_URL,
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        self.assertIsNone(args.model)
        self.assertEqual(
            args.max_coarse_duration,
            DEFAULT_MAX_COARSE_LINEUP_DURATION_SECONDS,
        )
        self.assertTrue(args.visual_ocr)
        self.assertEqual(args.ocr_python.name, "python")
        self.assertIsNone(args.output)
        option_names = {action.dest for action in parser._actions}
        self.assertNotIn("subchunk_duration", option_names)
        self.assertNotIn("workers", option_names)
        self.assertNotIn("refinement_dir", option_names)
        self.assertNotIn("scene_slideshow", option_names)

    def test_summarizes_lineup_request_usage_and_elapsed_time(self) -> None:
        summary = _request_summary(
            {
                "_validation_attempts": [
                    {
                        "attempt": 1,
                        "elapsed_seconds": 1.25,
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 20,
                            "total_tokens": 120,
                        },
                    },
                    {
                        "attempt": 2,
                        "elapsed_seconds": 0.75,
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 30,
                            "total_tokens": 130,
                        },
                    },
                ]
            }
        )

        self.assertEqual(summary["request_count"], 2)
        self.assertEqual(summary["elapsed_seconds"], 2.0)
        self.assertEqual(
            summary["usage"],
            {
                "prompt_tokens": 200,
                "completion_tokens": 50,
                "total_tokens": 250,
            },
        )

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
        self.assertIn("PySceneDetect", prompt)
        self.assertIn("start_anchor_text", prompt)
        self.assertIn("ngữ cảnh nhân sự mở rộng", prompt)
        self.assertIn("C reintroduced into midfield", prompt)
        self.assertIn("không được dài quá", prompt)
        self.assertIn("`75` giây", prompt)
        self.assertIn('"required": ["kickoff", "lineup_segments"]', prompt)
        self.assertIn("Trường `kickoff` là bắt buộc", prompt)
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

    def test_validated_kickoff_rejects_and_regenerates_post_kickoff_lineup(self) -> None:
        transcript = (
            TranscriptSegment(
                segment_id="chunk_000_seg_0001",
                start_seconds=0.0,
                end_seconds=60.0,
                text=(
                    "Belgium make ten changes. Vincent Kompany starts for the "
                    "first time in this tournament."
                ),
                source_chunk="chunk_000",
                chunk_start_seconds=0.0,
            ),
            TranscriptSegment(
                segment_id="chunk_002_seg_0001",
                start_seconds=120.0,
                end_seconds=180.0,
                text="Japan kick off and play the ball backwards.",
                source_chunk="chunk_002",
                chunk_start_seconds=120.0,
            ),
            TranscriptSegment(
                segment_id="chunk_006_seg_0001",
                start_seconds=360.0,
                end_seconds=420.0,
                text=(
                    "Belgium are playing with three central defenders and "
                    "Carrasco is the left wing back."
                ),
                source_chunk="chunk_006",
                chunk_start_seconds=360.0,
            ),
        )
        kickoff = {
            "evidence_segment_id": "chunk_002_seg_0001",
            "anchor_text": "Japan kick off and play the ball backwards",
            "confidence": 0.99,
        }
        wrong = {
            "kickoff": kickoff,
            "lineup_segments": [
                {
                    "team_name": "Belgium",
                    "start_seconds": 360.0,
                    "end_seconds": 420.0,
                    "confidence": 0.9,
                    "context": "lineup_presentation",
                    "evidence_segment_ids": ["chunk_006_seg_0001"],
                    "reason": "A tactical shape is mentioned after kickoff.",
                    "start_anchor_text": "Belgium are playing with three central defenders",
                    "end_anchor_text": "Carrasco is the left wing back",
                }
            ],
        }
        corrected = {
            "kickoff": kickoff,
            "lineup_segments": [
                {
                    "team_name": "Belgium",
                    "start_seconds": 0.0,
                    "end_seconds": 60.0,
                    "confidence": 0.96,
                    "context": "lineup_presentation",
                    "evidence_segment_ids": ["chunk_000_seg_0001"],
                    "reason": "The pre-kickoff team sheet is presented.",
                    "start_anchor_text": "Belgium make ten changes",
                    "end_anchor_text": "starts for the first time in this tournament",
                }
            ],
        }
        client = SequenceQwenClient([wrong, corrected])
        detector = QwenLineupDetector(
            model="qwen-test",
            client=client,
            expected_segment_count=1,
            max_attempts=2,
            require_kickoff_field=True,
        )

        result = detector.detect(
            transcript,
            video_name="WC_01.mp4",
            window_start_seconds=0.0,
            window_end_seconds=420.0,
        )

        self.assertEqual(len(client.calls), 2)
        self.assertEqual(result.segments[0].team_name, "Belgium")
        self.assertLess(result.segments[0].end_seconds, 120.0)
        self.assertEqual(
            result.raw_response["_kickoff_validation"]["status"],
            "validated",
        )

    def test_invalid_repair_keeps_the_best_valid_partial_response(self) -> None:
        transcript = (
            transcript_segment(
                "before",
                0.0,
                60.0,
                "Team A make five changes to their starting eleven.",
            ),
            transcript_segment(
                "kickoff",
                60.0,
                120.0,
                "The first half is now under way.",
            ),
            transcript_segment(
                "after",
                120.0,
                180.0,
                "Team B are playing with three central defenders.",
            ),
        )
        kickoff = {
            "evidence_segment_id": "kickoff",
            "anchor_text": "The first half is now under way",
            "confidence": 0.99,
        }
        valid_partial = {
            "kickoff": kickoff,
            "lineup_segments": [
                {
                    "team_name": "Team A",
                    "start_seconds": 0.0,
                    "end_seconds": 60.0,
                    "confidence": 0.9,
                    "context": "lineup_presentation",
                    "evidence_segment_ids": ["before"],
                    "reason": "A pre-kickoff lineup is presented.",
                    "start_anchor_text": "Team A make five changes",
                    "end_anchor_text": "to their starting eleven",
                },
                {
                    "team_name": "Team B",
                    "start_seconds": 120.0,
                    "end_seconds": 180.0,
                    "confidence": 0.8,
                    "context": "lineup_presentation",
                    "evidence_segment_ids": ["after"],
                    "reason": "A post-kickoff tactical shape is described.",
                    "start_anchor_text": "Team B are playing",
                    "end_anchor_text": "with three central defenders",
                },
            ],
        }
        invalid_repair = {
            "kickoff": kickoff,
            "lineup_segments": [
                {
                    **valid_partial["lineup_segments"][0],
                    "end_anchor_text": "words absent from the transcript",
                }
            ],
        }
        client = SequenceQwenClient([valid_partial, invalid_repair])
        detector = QwenLineupDetector(
            model="qwen-test",
            client=client,
            expected_segment_count=2,
            max_attempts=2,
            require_kickoff_field=True,
        )

        result = detector.detect(
            transcript,
            video_name="match.mp4",
            window_start_seconds=0.0,
            window_end_seconds=180.0,
        )

        self.assertEqual(len(client.calls), 2)
        self.assertEqual(
            [segment.team_name for segment in result.segments],
            ["Team A"],
        )
        self.assertEqual(result.status, "incomplete")
        self.assertEqual(
            result.raw_response["_partial_response_fallback"],
            "later_qwen_repairs_failed_validation",
        )

    def test_required_kickoff_field_cannot_be_omitted(self) -> None:
        with self.assertRaisesRegex(LineupDetectionError, "must contain kickoff"):
            parse_detection_result(
                {"lineup_segments": []},
                model="qwen-test",
                known_evidence=["s1"],
                evidence_bounds={"s1": (0.0, 60.0)},
                evidence_texts={"s1": "No kickoff yet."},
                window_start=0.0,
                window_end=60.0,
                require_kickoff_field=True,
            )

    def test_kickoff_recovers_a_uniquely_miscited_exact_anchor(self) -> None:
        result = parse_detection_result(
            {
                "kickoff": {
                    "evidence_segment_id": "after_kickoff",
                    "anchor_text": "the first half is now under way",
                    "confidence": 0.95,
                },
                "lineup_segments": [],
            },
            model="qwen-test",
            known_evidence=["kickoff", "after_kickoff"],
            evidence_bounds={
                "kickoff": (60.0, 120.0),
                "after_kickoff": (120.0, 180.0),
            },
            evidence_texts={
                "kickoff": "The first half is now under way.",
                "after_kickoff": "The home side keep the ball.",
            },
            window_start=0.0,
            window_end=180.0,
            require_kickoff_field=True,
        )

        validation = result.raw_response["_kickoff_validation"]
        self.assertEqual(validation["evidence_segment_id"], "kickoff")
        self.assertEqual(
            validation["evidence_segment_id_repaired_from"],
            "after_kickoff",
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

    def test_rejects_candidate_spanning_a_coming_out_ceremony(self) -> None:
        text = (
            "A preliminary lineup fragment. Players will be coming out. "
            "Players coming through the line. Frontale coach makes three "
            "changes to the starting eleven."
        )
        transcript = (
            transcript_segment("chunk_001_seg_0001", 300.0, 360.0, text),
        )
        client = FakeQwenClient(
            {
                "lineup_segments": [
                    {
                        "team_name": "Frontale",
                        "start_seconds": 300.0,
                        "end_seconds": 360.0,
                        "confidence": 0.9,
                        "context": "lineup_presentation",
                        "evidence_segment_ids": ["chunk_001_seg_0001"],
                        "start_anchor_text": (
                            "A preliminary lineup fragment"
                        ),
                        "end_anchor_text": (
                            "makes three changes to the starting eleven"
                        ),
                        "reason": "The model merged ceremony and roster.",
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

        self.assertEqual(result.segments, tuple())
        self.assertEqual(len(result.raw_response["_rejected_candidates"]), 1)

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
