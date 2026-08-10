"""Gemini adapter for timestamped audio transcription."""

from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from .schema import TranscriptSegment, TranscriptionResult


DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_PROMPT_PATH = Path(__file__).parent / "prompts" / "transcribe_prompt.txt"
MAX_TIMESTAMP_OVERRUN_SECONDS = 15.0
TRANSCRIPT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_seconds": {
                        "type": "number",
                        "minimum": 0,
                        "description": "Start time relative to the audio file.",
                    },
                    "end_seconds": {
                        "type": "number",
                        "minimum": 0,
                        "description": "End time relative to the audio file.",
                    },
                    "text": {"type": "string"},
                },
                "required": ["start_seconds", "end_seconds", "text"],
            },
        }
    },
    "required": ["segments"],
}


class GeminiTranscriptionError(RuntimeError):
    """Raised when Gemini cannot produce a valid timestamped transcript."""


def load_prompt(
    language: str,
    prompt_path: Path = DEFAULT_PROMPT_PATH,
    *,
    audio_duration_seconds: float | None = None,
) -> str:
    if not prompt_path.is_file():
        raise GeminiTranscriptionError(f"Transcription prompt does not exist: {prompt_path}")
    template = prompt_path.read_text(encoding="utf-8")
    duration = (
        f"{audio_duration_seconds:.3f}"
        if audio_duration_seconds is not None
        else "unknown"
    )
    return template.replace("{LANGUAGE}", language.strip() or "auto").replace(
        "{AUDIO_DURATION_SECONDS}", duration
    )


def build_transcript_response_schema(audio_duration_seconds: float) -> dict[str, Any]:
    schema = deepcopy(TRANSCRIPT_RESPONSE_SCHEMA)
    properties = schema["properties"]["segments"]["items"]["properties"]
    properties["start_seconds"]["maximum"] = audio_duration_seconds
    properties["end_seconds"]["maximum"] = audio_duration_seconds
    return schema


def _file_state_name(uploaded_file: object) -> str:
    state = getattr(uploaded_file, "state", None)
    if state is None:
        return "ACTIVE"
    name = getattr(state, "name", None)
    return str(name or state).upper()


class GeminiTranscriber:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        client: object | None = None,
        poll_interval_seconds: float = 1.0,
        file_ready_timeout_seconds: float = 60.0,
    ) -> None:
        if not model.strip():
            raise GeminiTranscriptionError("Gemini model cannot be empty.")
        if poll_interval_seconds <= 0 or file_ready_timeout_seconds <= 0:
            raise GeminiTranscriptionError("Gemini polling durations must be positive.")
        self.model = model
        self.poll_interval_seconds = poll_interval_seconds
        self.file_ready_timeout_seconds = file_ready_timeout_seconds
        if client is not None:
            self.client = client
            return
        if not api_key:
            raise GeminiTranscriptionError(
                "Missing GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
            )
        try:
            from google import genai
        except ModuleNotFoundError as exc:
            raise GeminiTranscriptionError(
                "google-genai is not installed. Install requirements.txt first."
            ) from exc
        self.client = genai.Client(api_key=api_key)

    def transcribe(
        self,
        audio_path: Path,
        *,
        audio_duration_seconds: float,
        language: str = "auto",
        source_chunk: str = "chunk_000",
        chunk_start_seconds: float = 0.0,
        prompt_path: Path = DEFAULT_PROMPT_PATH,
        raw_response_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> TranscriptionResult:
        audio = audio_path.expanduser().resolve()
        if not audio.is_file():
            raise GeminiTranscriptionError(f"Audio file does not exist: {audio}")
        if not math.isfinite(audio_duration_seconds) or audio_duration_seconds <= 0:
            raise GeminiTranscriptionError("audio_duration_seconds must be positive.")
        if not math.isfinite(chunk_start_seconds) or chunk_start_seconds < 0:
            raise GeminiTranscriptionError("chunk_start_seconds must be non-negative.")
        if not source_chunk.strip():
            raise GeminiTranscriptionError("source_chunk cannot be empty.")

        prompt = load_prompt(
            language,
            prompt_path,
            audio_duration_seconds=audio_duration_seconds,
        )
        uploaded_file: object | None = None
        uploaded_name: str | None = None
        try:
            uploaded_file = self.client.files.upload(
                file=str(audio),
                config={"mime_type": "audio/wav"},
            )
            uploaded_name = getattr(uploaded_file, "name", None)
            uploaded_file = self._wait_until_active(uploaded_file, uploaded_name)
            response = self.client.models.generate_content(
                model=self.model,
                contents=[uploaded_file, prompt],
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": build_transcript_response_schema(
                        audio_duration_seconds
                    ),
                },
            )
            raw_text = getattr(response, "text", None)
            if not isinstance(raw_text, str) or not raw_text.strip():
                raise GeminiTranscriptionError("Gemini returned an empty response.")
            try:
                payload = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                raise GeminiTranscriptionError(
                    "Gemini transcription response is not valid JSON."
                ) from exc
            if raw_response_callback is not None and isinstance(payload, dict):
                raw_response_callback(payload)
            return self._parse_payload(
                payload,
                audio_duration_seconds=audio_duration_seconds,
                source_chunk=source_chunk,
                chunk_start_seconds=chunk_start_seconds,
            )
        except GeminiTranscriptionError:
            raise
        except Exception as exc:
            raise GeminiTranscriptionError(
                f"Gemini transcription request failed: {exc}"
            ) from exc
        finally:
            if uploaded_name:
                try:
                    self.client.files.delete(name=uploaded_name)
                except Exception:
                    pass

    def _wait_until_active(self, uploaded_file: object, uploaded_name: str | None) -> object:
        state = _file_state_name(uploaded_file)
        if "FAILED" in state:
            raise GeminiTranscriptionError("Gemini rejected the uploaded audio file.")
        if "ACTIVE" in state or not uploaded_name:
            return uploaded_file

        deadline = time.monotonic() + self.file_ready_timeout_seconds
        current = uploaded_file
        while time.monotonic() < deadline:
            time.sleep(self.poll_interval_seconds)
            current = self.client.files.get(name=uploaded_name)
            state = _file_state_name(current)
            if "ACTIVE" in state:
                return current
            if "FAILED" in state:
                raise GeminiTranscriptionError(
                    "Gemini failed while processing the uploaded audio file."
                )
        raise GeminiTranscriptionError(
            "Timed out waiting for the uploaded Gemini audio file to become active."
        )

    def _parse_payload(
        self,
        payload: object,
        *,
        audio_duration_seconds: float,
        source_chunk: str,
        chunk_start_seconds: float,
    ) -> TranscriptionResult:
        if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
            raise GeminiTranscriptionError(
                "Gemini response must contain a segments array."
            )

        parsed: list[tuple[float, float, str]] = []
        for index, item in enumerate(payload["segments"], start=1):
            if not isinstance(item, dict):
                raise GeminiTranscriptionError(
                    f"Transcript segment {index} must be an object."
                )
            try:
                relative_start = float(item["start_seconds"])
                relative_end = float(item["end_seconds"])
            except (KeyError, TypeError, ValueError) as exc:
                raise GeminiTranscriptionError(
                    f"Transcript segment {index} has invalid timestamps."
                ) from exc
            text = str(item.get("text", "")).strip()
            if not text:
                raise GeminiTranscriptionError(
                    f"Transcript segment {index} has empty text."
                )
            if (
                not math.isfinite(relative_start)
                or not math.isfinite(relative_end)
                or relative_start < 0
                or relative_end <= relative_start
            ):
                raise GeminiTranscriptionError(
                    f"Transcript segment {index} has an invalid time range."
                )
            if relative_start >= audio_duration_seconds:
                raise GeminiTranscriptionError(
                    f"Transcript segment {index} starts outside the audio duration "
                    f"({relative_start:.3f}s >= {audio_duration_seconds:.3f}s)."
                )
            if relative_end > audio_duration_seconds + MAX_TIMESTAMP_OVERRUN_SECONDS:
                raise GeminiTranscriptionError(
                    f"Transcript segment {index} ends too far outside the audio duration "
                    f"({relative_end:.3f}s > {audio_duration_seconds:.3f}s)."
                )
            relative_end = min(relative_end, audio_duration_seconds)
            parsed.append((relative_start, relative_end, text))

        parsed.sort(key=lambda row: (row[0], row[1]))
        segments = tuple(
            TranscriptSegment(
                segment_id=f"{source_chunk}_seg_{index:04d}",
                start_seconds=chunk_start_seconds + relative_start,
                end_seconds=chunk_start_seconds + relative_end,
                text=text,
                source_chunk=source_chunk,
                chunk_start_seconds=chunk_start_seconds,
            )
            for index, (relative_start, relative_end, text) in enumerate(
                parsed, start=1
            )
        )
        result = TranscriptionResult(
            model=self.model,
            segments=segments,
            raw_response=payload,
        )
        result.validate()
        return result
