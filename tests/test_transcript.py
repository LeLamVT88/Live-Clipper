from __future__ import annotations

import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from transcript.audio import (
    AudioExtractionError,
    MediaInfo,
    plan_audio_chunks,
    probe_media,
    wav_duration_seconds,
)
from transcript.qwen import QwenTranscriber, QwenTranscriptionError
from transcript.schema import (
    TranscriptSegment,
    TranscriptValidationError,
    read_jsonl,
    write_json,
    write_jsonl,
)
from transcript.transcribe_video import (
    build_parser as build_transcription_parser,
    fit_processing_duration,
    load_cached_chunk,
)


class FakeQwenClient:
    def __init__(
        self,
        text: str,
        *,
        failures_before_success: int = 0,
        status_code: int = 200,
    ) -> None:
        self.text = text
        self.failures_before_success = failures_before_success
        self.status_code = status_code
        self.calls: list[dict[str, object]] = []

    def call(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if len(self.calls) <= self.failures_before_success:
            raise RuntimeError("503 UNAVAILABLE: model is overloaded")
        return SimpleNamespace(
            status_code=self.status_code,
            code="InvalidParameter" if self.status_code != 200 else None,
            message="bad request" if self.status_code != 200 else None,
            request_id="request-test",
            usage={"seconds": 20},
            output={
                "choices": [
                    {
                        "message": {
                            "content": [{"text": self.text}],
                            "annotations": [
                                {
                                    "type": "audio_info",
                                    "language": "vi",
                                    "emotion": "neutral",
                                }
                            ],
                        }
                    }
                ]
            },
        )


class TranscriptSchemaTests(unittest.TestCase):
    def test_cli_uses_ten_one_minute_chunks_and_exposes_resume(self) -> None:
        args = build_transcription_parser().parse_args(
            ["--video", "match.mp4", "--resume"]
        )

        self.assertEqual(args.duration, "600")
        self.assertEqual(args.chunk_duration, "60")
        self.assertEqual(args.max_chunks, 10)
        self.assertEqual(args.workers, 4)
        self.assertTrue(args.resume)

    def test_plans_fixed_chunks_covering_the_complete_requested_window(self) -> None:
        chunks = plan_audio_chunks(
            start_seconds=10.0,
            duration_seconds=620.0,
            chunk_duration_seconds=300.0,
        )

        self.assertEqual(
            [
                (chunk.chunk_id, chunk.start_seconds, chunk.duration_seconds)
                for chunk in chunks
            ],
            [
                ("chunk_000", 10.0, 300.0),
                ("chunk_001", 310.0, 300.0),
                ("chunk_002", 610.0, 20.0),
            ],
        )

    def test_chunk_planner_rejects_invalid_duration(self) -> None:
        with self.assertRaises(AudioExtractionError):
            plan_audio_chunks(
                start_seconds=0.0,
                duration_seconds=600.0,
                chunk_duration_seconds=0.0,
            )

    def test_short_video_uses_fewer_than_ten_chunks(self) -> None:
        media = MediaInfo(
            duration_seconds=354.2,
            audio_stream_count=1,
            video_stream_count=1,
        )
        duration = fit_processing_duration(
            media=media,
            start_seconds=0.0,
            requested_duration_seconds=600.0,
            chunk_duration_seconds=60.0,
            max_chunks=10,
        )
        chunks = plan_audio_chunks(
            start_seconds=0.0,
            duration_seconds=duration,
            chunk_duration_seconds=60.0,
        )

        self.assertEqual(duration, 354.2)
        self.assertEqual(len(chunks), 6)
        self.assertAlmostEqual(chunks[-1].duration_seconds, 54.2)

    def test_processing_window_never_exceeds_ten_chunks(self) -> None:
        media = MediaInfo(
            duration_seconds=1_876.0,
            audio_stream_count=1,
            video_stream_count=1,
        )

        duration = fit_processing_duration(
            media=media,
            start_seconds=0.0,
            requested_duration_seconds=1_200.0,
            chunk_duration_seconds=60.0,
            max_chunks=10,
        )

        self.assertEqual(duration, 600.0)

    def test_processing_rejects_input_without_audio(self) -> None:
        media = MediaInfo(
            duration_seconds=600.0,
            audio_stream_count=0,
            video_stream_count=1,
        )

        with self.assertRaisesRegex(AudioExtractionError, "no audio stream"):
            fit_processing_duration(
                media=media,
                start_seconds=0.0,
                requested_duration_seconds=600.0,
                chunk_duration_seconds=60.0,
                max_chunks=10,
            )

    def test_ffprobe_preflight_reads_duration_and_streams(self) -> None:
        response = SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [
                        {"codec_type": "video", "duration": "99.9"},
                        {"codec_type": "audio", "duration": "99.8"},
                    ],
                    "format": {"duration": "100.0"},
                }
            ),
            stderr="",
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.mp4"
            source.write_bytes(b"input")
            with patch("transcript.audio.subprocess.run", return_value=response):
                info = probe_media(source)

        self.assertEqual(info.duration_seconds, 100.0)
        self.assertEqual(info.audio_stream_count, 1)
        self.assertEqual(info.video_stream_count, 1)

    def test_segment_serializes_normalized_values(self) -> None:
        segment = TranscriptSegment(
            segment_id="chunk_000_seg_0001",
            start_seconds=12.34567,
            end_seconds=15.78912,
            text="  Đội hình xuất phát  ",
            source_chunk="chunk_000",
            chunk_start_seconds=10.0,
        )

        self.assertEqual(
            segment.to_dict(),
            {
                "segment_id": "chunk_000_seg_0001",
                "start_seconds": 12.346,
                "end_seconds": 15.789,
                "text": "Đội hình xuất phát",
                "source_chunk": "chunk_000",
                "chunk_start_seconds": 10.0,
            },
        )

    def test_writer_rejects_duplicates_before_overwriting_output(self) -> None:
        duplicate = TranscriptSegment(
            segment_id="duplicate",
            start_seconds=0.0,
            end_seconds=1.0,
            text="Một",
            source_chunk="chunk_000",
            chunk_start_seconds=0.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "transcript.jsonl"
            output.write_text("existing\n", encoding="utf-8")

            with self.assertRaises(TranscriptValidationError):
                write_jsonl((duplicate, duplicate), output)

            self.assertEqual(output.read_text(encoding="utf-8"), "existing\n")

    def test_wav_duration_uses_header_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "sample.wav"
            with wave.open(str(audio_path), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(16_000)
                audio.writeframes(b"\x00\x00" * 24_000)

            self.assertEqual(wav_duration_seconds(audio_path), 1.5)

    def test_reads_persisted_transcript_jsonl(self) -> None:
        segment = TranscriptSegment(
            segment_id="chunk_001_seg_0001",
            start_seconds=301.0,
            end_seconds=304.0,
            text="Starting eleven",
            source_chunk="chunk_001",
            chunk_start_seconds=300.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            transcript_path = Path(directory) / "transcript.jsonl"
            write_jsonl((segment,), transcript_path)

            loaded = read_jsonl(transcript_path)

        self.assertEqual(loaded, (segment,))

    def test_loads_a_complete_chunk_checkpoint(self) -> None:
        segment = TranscriptSegment(
            segment_id="chunk_001_seg_0001",
            start_seconds=121.0,
            end_seconds=124.0,
            text="Starting eleven",
            source_chunk="chunk_001",
            chunk_start_seconds=120.0,
        )
        payload = {
            "segments": [
                {
                    "start_seconds": 1.0,
                    "end_seconds": 4.0,
                    "text": "Starting eleven",
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript_path = root / "chunk.jsonl"
            raw_path = root / "chunk.json"
            write_jsonl((segment,), transcript_path)
            write_json(payload, raw_path)

            segments, raw = load_cached_chunk(
                transcript_path=transcript_path,
                raw_response_path=raw_path,
                source_chunk="chunk_001",
                chunk_start_seconds=120.0,
            )

        self.assertEqual(segments, (segment,))
        self.assertEqual(raw, payload)


class QwenTranscriberTests(unittest.TestCase):
    def _audio_file(self, directory: str) -> Path:
        path = Path(directory) / "chunk.wav"
        path.write_bytes(b"fake wav content")
        return path

    def test_transcription_uses_chunk_bounds_and_adds_video_offset(self) -> None:
        client = FakeQwenClient("Đây là đội hình xuất phát của PSG.")
        transcriber = QwenTranscriber(model="qwen-test", client=client)
        captured: list[dict[str, object]] = []

        with tempfile.TemporaryDirectory() as directory:
            audio_path = self._audio_file(directory)
            result = transcriber.transcribe(
                audio_path,
                audio_duration_seconds=20.0,
                language="vi",
                source_chunk="chunk_003",
                chunk_start_seconds=300.0,
                raw_response_callback=captured.append,
            )

        self.assertEqual(len(result.segments), 1)
        segment = result.segments[0]
        self.assertEqual(segment.segment_id, "chunk_003_seg_0001")
        self.assertEqual(segment.start_seconds, 300.0)
        self.assertEqual(segment.end_seconds, 320.0)
        self.assertEqual(segment.text, "Đây là đội hình xuất phát của PSG.")
        call = client.calls[0]
        self.assertEqual(call["model"], "qwen-test")
        self.assertEqual(call["result_format"], "message")
        self.assertEqual(
            call["asr_options"],
            {"enable_itn": True, "language": "vi"},
        )
        messages = call["messages"]
        self.assertEqual(
            messages[0]["content"][0]["audio"],
            str(audio_path.resolve()),
        )
        self.assertEqual(captured[0]["request_id"], "request-test")
        self.assertEqual(captured[0]["segments"][0]["end_seconds"], 20.0)

    def test_auto_language_is_omitted_and_context_is_optional(self) -> None:
        client = FakeQwenClient("Starting eleven")
        transcriber = QwenTranscriber(model="qwen-test", client=client)

        with tempfile.TemporaryDirectory() as directory:
            transcriber.transcribe(
                self._audio_file(directory),
                audio_duration_seconds=10.0,
                context="PSG versus Aston Villa; football commentary.",
            )

        call = client.calls[0]
        self.assertEqual(call["asr_options"], {"enable_itn": True})
        messages = call["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("PSG", messages[0]["content"][0]["text"])
        self.assertEqual(messages[1]["role"], "user")

    def test_empty_asr_text_is_treated_as_silence(self) -> None:
        client = FakeQwenClient("   ")
        transcriber = QwenTranscriber(model="qwen-test", client=client)
        captured: list[dict[str, object]] = []

        with tempfile.TemporaryDirectory() as directory:
            result = transcriber.transcribe(
                self._audio_file(directory),
                audio_duration_seconds=10.0,
                raw_response_callback=captured.append,
            )

        self.assertEqual(result.segments, ())
        self.assertEqual(captured[0]["segments"], [])

    def test_retries_a_transient_request_without_restarting_the_run(self) -> None:
        client = FakeQwenClient("Lineup", failures_before_success=1)
        transcriber = QwenTranscriber(
            model="qwen-test",
            client=client,
            max_attempts=2,
            retry_delay_seconds=0,
        )

        with tempfile.TemporaryDirectory() as directory:
            result = transcriber.transcribe(
                self._audio_file(directory),
                audio_duration_seconds=10.0,
            )

        self.assertEqual(len(result.segments), 1)
        self.assertEqual(len(client.calls), 2)

    def test_non_successful_dashscope_response_is_rejected(self) -> None:
        client = FakeQwenClient("", status_code=400)
        transcriber = QwenTranscriber(
            model="qwen-test",
            client=client,
            max_attempts=1,
        )

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(QwenTranscriptionError, "400"):
                transcriber.transcribe(
                    self._audio_file(directory),
                    audio_duration_seconds=10.0,
                )


if __name__ == "__main__":
    unittest.main()
