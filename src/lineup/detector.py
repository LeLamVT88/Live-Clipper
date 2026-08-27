"""Qwen adapter for transcript-driven lineup detection."""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from transcript.qwen import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_RETRY_DELAY_SECONDS,
    QwenTranscriptionError,
    ensure_qwen_success,
    is_transient_qwen_error,
    qwen_message,
    qwen_message_text,
)
from transcript.schema import TranscriptSegment

from .schema import (
    LINEUP_RESPONSE_SCHEMA,
    LineupDetectionError,
    LineupDetectionResult,
    LineupSegment,
    parse_detection_result,
    write_lineup_csv,
)


DEFAULT_PROMPT_PATH = Path(__file__).parent / "prompts" / "detect_lineup_prompt.txt"
DEFAULT_TEXT_MODEL = "qwen-plus"


class QwenLineupRequestError(LineupDetectionError):
    """A request-level failure, kept separate from payload validation errors."""


def build_detection_prompt(
    segments: Iterable[TranscriptSegment],
    *,
    video_name: str,
    window_start_seconds: float,
    window_end_seconds: float,
    prompt_path: Path = DEFAULT_PROMPT_PATH,
) -> str:
    if not prompt_path.is_file():
        raise LineupDetectionError(f"Lineup prompt does not exist: {prompt_path}")
    transcript = [
        {
            "segment_id": segment.segment_id,
            "start_seconds": segment.start_seconds,
            "end_seconds": segment.end_seconds,
            "text": segment.text,
        }
        for segment in segments
    ]
    values = {
        "{VIDEO_NAME}": video_name,
        "{WINDOW_START_SECONDS}": f"{window_start_seconds:.3f}",
        "{WINDOW_END_SECONDS}": f"{window_end_seconds:.3f}",
        "{TRANSCRIPT_JSON}": json.dumps(transcript, ensure_ascii=False),
        "{LINEUP_JSON_SCHEMA}": json.dumps(
            LINEUP_RESPONSE_SCHEMA, ensure_ascii=False
        ),
    }
    prompt = prompt_path.read_text(encoding="utf-8")
    for placeholder, value in values.items():
        prompt = prompt.replace(placeholder, value)
    return prompt


def _response_payload(response: object) -> dict[str, object]:
    try:
        message = qwen_message(response, operation="Qwen lineup detection")
    except QwenTranscriptionError as exc:
        raise LineupDetectionError(str(exc)) from exc
    raw_text = qwen_message_text(message)
    if not raw_text:
        raise LineupDetectionError("Qwen returned an empty lineup response.")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise LineupDetectionError(
            "Qwen lineup response is not valid JSON."
        ) from exc
    if isinstance(payload, list):
        payload = {"lineup_segments": payload}
    if not isinstance(payload, dict):
        raise LineupDetectionError(
            "Qwen lineup response must be an object or candidate array."
        )
    return payload


class QwenLineupDetector:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_TEXT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        client: object | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
        expected_segment_count: int | None = None,
    ) -> None:
        if not model.strip():
            raise LineupDetectionError("Qwen text model cannot be empty.")
        if not base_url.strip():
            raise LineupDetectionError("DashScope base URL cannot be empty.")
        if max_attempts <= 0 or retry_delay_seconds < 0:
            raise LineupDetectionError(
                "Qwen retry attempts must be positive and delay non-negative."
            )
        if expected_segment_count is not None and expected_segment_count <= 0:
            raise LineupDetectionError(
                "Expected lineup segment count must be positive."
            )
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.expected_segment_count = expected_segment_count
        self.client = (
            client
            if client is not None
            else self._create_client(api_key, base_url)
        )

    @staticmethod
    def _create_client(api_key: str | None, base_url: str) -> object:
        if not api_key:
            raise LineupDetectionError(
                "Missing DASHSCOPE_API_KEY environment variable."
            )
        try:
            import dashscope
        except ModuleNotFoundError as exc:
            raise LineupDetectionError(
                "dashscope is not installed. Install requirements.txt first."
            ) from exc
        dashscope.base_http_api_url = base_url
        return dashscope.Generation

    def detect(
        self,
        segments: Iterable[TranscriptSegment],
        *,
        video_name: str,
        window_start_seconds: float,
        window_end_seconds: float,
        prompt_path: Path = DEFAULT_PROMPT_PATH,
    ) -> LineupDetectionResult:
        transcript = tuple(segments)
        self._validate_input(video_name, window_start_seconds, window_end_seconds)
        if not transcript:
            return LineupDetectionResult(
                self.model, (), {"lineup_segments": []}, status="empty"
            )

        prompt = build_detection_prompt(
            transcript,
            video_name=video_name,
            window_start_seconds=window_start_seconds,
            window_end_seconds=window_end_seconds,
            prompt_path=prompt_path,
        )
        evidence_bounds = {
            segment.segment_id: (
                segment.start_seconds,
                segment.end_seconds,
            )
            for segment in transcript
        }
        evidence_texts = {
            segment.segment_id: segment.text for segment in transcript
        }
        last_error: LineupDetectionError | None = None
        attempt_log: list[dict[str, object]] = []
        for attempt in range(1, self.max_attempts + 1):
            payload: dict[str, object] | None = None
            try:
                payload = _response_payload(self._generate(prompt))
                result = parse_detection_result(
                    payload,
                    model=self.model,
                    known_evidence=evidence_bounds,
                    evidence_bounds=evidence_bounds,
                    evidence_texts=evidence_texts,
                    window_start=window_start_seconds,
                    window_end=window_end_seconds,
                )
                count = len(result.segments)
                rejected_candidates = result.raw_response.get(
                    "_rejected_candidates", []
                )
                if (
                    rejected_candidates
                    and self.expected_segment_count is not None
                    and count < self.expected_segment_count
                    and attempt < self.max_attempts
                ):
                    raise LineupDetectionError(
                        "One or more Qwen lineup candidates were rejected as "
                        "player-walkout/ceremony content."
                    )
                if count == 0:
                    status = "empty"
                elif self.expected_segment_count is None:
                    status = "unconstrained"
                elif count == self.expected_segment_count:
                    status = "complete"
                else:
                    status = "incomplete"
                raw_response = dict(result.raw_response)
                raw_response["_validation_attempts"] = attempt_log + [
                    {"attempt": attempt, "status": "accepted"}
                ]
                completed = replace(
                    result,
                    raw_response=raw_response,
                    status=status,
                )
                completed.validate()
                return completed
            except QwenLineupRequestError as exc:
                last_error = exc
                attempt_log.append(
                    {
                        "attempt": attempt,
                        "status": "request_error",
                        "error": str(exc),
                    }
                )
                if (
                    attempt >= self.max_attempts
                    or not is_transient_qwen_error(exc)
                ):
                    raise
                delay = self.retry_delay_seconds * (2 ** (attempt - 1))
                print(
                    "Transient Qwen lineup detection error; retrying in "
                    f"{delay:g}s ({attempt + 1}/{self.max_attempts})."
                )
                time.sleep(delay)
            except LineupDetectionError as exc:
                last_error = exc
                attempt_log.append(
                    {
                        "attempt": attempt,
                        "status": "invalid_payload",
                        "error": str(exc),
                        "payload": payload,
                    }
                )
                if attempt >= self.max_attempts:
                    raise
                print(
                    "Invalid Qwen lineup payload; regenerating "
                    f"({attempt + 1}/{self.max_attempts}): {exc}"
                )
        assert last_error is not None
        raise last_error

    @staticmethod
    def _validate_input(video_name: str, start: float, end: float) -> None:
        if not video_name.strip():
            raise LineupDetectionError("video_name cannot be empty.")
        valid_window = (
            all(math.isfinite(value) for value in (start, end))
            and start >= 0
            and end > start
        )
        if not valid_window:
            raise LineupDetectionError("Invalid transcription window.")

    def _generate(self, prompt: str) -> object:
        call_kwargs: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "result_format": "message",
            "response_format": {"type": "json_object"},
        }
        if self.api_key:
            call_kwargs["api_key"] = self.api_key

        try:
            response = self.client.call(**call_kwargs)  # type: ignore[attr-defined]
            ensure_qwen_success(response, operation="Qwen lineup detection")
            return response
        except Exception as exc:
            raise QwenLineupRequestError(
                f"Qwen lineup detection request failed: {exc}"
            ) from exc
