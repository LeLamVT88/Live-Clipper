from __future__ import annotations

import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from transcript.audio import AudioExtractionError, plan_audio_chunks, wav_duration_seconds
from transcript.gemini import GeminiTranscriber, GeminiTranscriptionError
from transcript.schema import (
    TranscriptSegment,
    TranscriptValidationError,
    read_jsonl,
    write_jsonl,
)


class FakeFiles:
    def __init__(self) -> None:
        self.upload_calls: list[dict[str, object]] = []
        self.deleted_names: list[str] = []

    def upload(self, **kwargs: object) -> object:
        self.upload_calls.append(kwargs)
        return SimpleNamespace(name="files/test-audio", state="ACTIVE")

    def get(self, *, name: str) -> object:
        return SimpleNamespace(name=name, state="ACTIVE")

    def delete(self, *, name: str) -> None:
        self.deleted_names.append(name)


class FakeModels:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return SimpleNamespace(text=json.dumps(self.payload, ensure_ascii=False))


class FakeGeminiClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.files = FakeFiles()
        self.models = FakeModels(payload)


class TranscriptSchemaTests(unittest.TestCase):
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


class GeminiTranscriberTests(unittest.TestCase):
    def _audio_file(self, directory: str) -> Path:
        path = Path(directory) / "chunk.wav"
        path.write_bytes(b"fake wav content")
        return path

    def test_transcription_orders_segments_and_adds_video_offset(self) -> None:
        client = FakeGeminiClient(
            {
                "segments": [
                    {"start_seconds": 8, "end_seconds": 12, "text": "Cầu thủ B"},
                    {"start_seconds": 1.25, "end_seconds": 4.5, "text": "Cầu thủ A"},
                ]
            }
        )
        transcriber = GeminiTranscriber(model="gemini-test", client=client)

        with tempfile.TemporaryDirectory() as directory:
            result = transcriber.transcribe(
                self._audio_file(directory),
                audio_duration_seconds=20.0,
                language="vi",
                source_chunk="chunk_003",
                chunk_start_seconds=300.0,
            )

        self.assertEqual(
            [segment.segment_id for segment in result.segments],
            ["chunk_003_seg_0001", "chunk_003_seg_0002"],
        )
        self.assertEqual(
            [segment.start_seconds for segment in result.segments],
            [301.25, 308.0],
        )
        self.assertEqual(client.files.deleted_names, ["files/test-audio"])
        self.assertEqual(client.files.upload_calls[0]["config"], {"mime_type": "audio/wav"})
        generation_call = client.models.calls[0]
        self.assertEqual(generation_call["model"], "gemini-test")
        config = generation_call["config"]
        self.assertEqual(config["response_mime_type"], "application/json")
        self.assertIn("response_json_schema", config)
        schema = config["response_json_schema"]
        properties = schema["properties"]["segments"]["items"]["properties"]
        self.assertEqual(properties["end_seconds"]["maximum"], 20.0)
        self.assertNotIn("temperature", config)

    def test_invalid_model_timestamp_still_deletes_uploaded_file(self) -> None:
        client = FakeGeminiClient(
            {
                "segments": [
                    {"start_seconds": 1, "end_seconds": 30, "text": "Ngoài audio"}
                ]
            }
        )
        transcriber = GeminiTranscriber(model="gemini-test", client=client)

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                GeminiTranscriptionError, "outside the audio duration"
            ):
                transcriber.transcribe(
                    self._audio_file(directory),
                    audio_duration_seconds=10.0,
                )

        self.assertEqual(client.files.deleted_names, ["files/test-audio"])

    def test_small_timestamp_overrun_is_clamped_and_raw_response_is_saved(self) -> None:
        payload = {
            "segments": [
                {"start_seconds": 295, "end_seconds": 305, "text": "Cuối chunk"}
            ]
        }
        client = FakeGeminiClient(payload)
        transcriber = GeminiTranscriber(model="gemini-test", client=client)
        captured: list[dict[str, object]] = []

        with tempfile.TemporaryDirectory() as directory:
            result = transcriber.transcribe(
                self._audio_file(directory),
                audio_duration_seconds=300.0,
                raw_response_callback=captured.append,
            )

        self.assertEqual(result.segments[0].end_seconds, 300.0)
        self.assertEqual(captured, [payload])


if __name__ == "__main__":
    unittest.main()
