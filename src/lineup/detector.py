"""Qwen text-over-transcript adapter for coarse lineup detection."""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from transcript.qwen import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_RETRY_DELAY_SECONDS,
    QwenTranscriptionError,
    ensure_qwen_success,
    is_transient_qwen_error,
    qwen_message,
    qwen_message_text,
    response_field,
    response_to_plain,
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
DEFAULT_TEXT_BASE_URL = (
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_COARSE_LINEUP_DURATION_SECONDS = 75.0


class QwenLineupRequestError(LineupDetectionError):
    """A request-level failure, kept separate from payload validation errors."""


class QwenLineupValidationError(LineupDetectionError):
    """All model payloads failed validation, with diagnostics preserved."""

    def __init__(
        self,
        message: str,
        attempts: Iterable[dict[str, object]],
    ) -> None:
        super().__init__(message)
        self.attempts = tuple(dict(attempt) for attempt in attempts)

    def raw_response(self) -> dict[str, object]:
        return {
            "lineup_segments": [],
            "_error": str(self),
            "_validation_attempts": list(self.attempts),
        }


class OpenAICompatibleLineupClient:
    """Small dependency-free client for an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        opener: Callable[..., object] = urlopen,
    ) -> None:
        normalized_url = base_url.rstrip("/")
        if not normalized_url:
            raise LineupDetectionError("Lineup model base URL cannot be empty.")
        if timeout_seconds <= 0:
            raise LineupDetectionError("Lineup request timeout must be positive.")
        self.endpoint = f"{normalized_url}/chat/completions"
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.opener = opener

    def call(self, **kwargs: object) -> object:
        request_started = time.monotonic()
        body = {
            key: kwargs[key]
            for key in (
                "model",
                "messages",
                "temperature",
                "max_tokens",
                "response_format",
                "enable_thinking",
            )
            if key in kwargs
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            self.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with self.opener(  # type: ignore[attr-defined]
                request, timeout=self.timeout_seconds
            ) as response:
                status = int(getattr(response, "status", 200))
                raw_payload = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(
                f"Lineup model HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Lineup model connection failed: {exc.reason}"
            ) from exc
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Lineup model returned invalid response JSON.") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Lineup model response must be a JSON object.")
        return {
            "status_code": status,
            "request_id": payload.get("id"),
            "output": {"choices": payload.get("choices", [])},
            "usage": payload.get("usage"),
            "elapsed_seconds": round(time.monotonic() - request_started, 3),
        }


def build_detection_prompt(
    segments: Iterable[TranscriptSegment],
    *,
    video_name: str,
    window_start_seconds: float,
    window_end_seconds: float,
    max_coarse_duration_seconds: float = (
        DEFAULT_MAX_COARSE_LINEUP_DURATION_SECONDS
    ),
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
        "{MAX_COARSE_LINEUP_DURATION_SECONDS}": (
            f"{max_coarse_duration_seconds:g}"
        ),
        "{TRANSCRIPT_JSON}": json.dumps(transcript, ensure_ascii=False),
        "{LINEUP_JSON_SCHEMA}": json.dumps(
            LINEUP_RESPONSE_SCHEMA, ensure_ascii=False
        ),
    }
    prompt = prompt_path.read_text(encoding="utf-8")
    for placeholder, value in values.items():
        prompt = prompt.replace(placeholder, value)
    return prompt


def _repair_detection_prompt(
    base_prompt: str,
    *,
    error: LineupDetectionError,
    payload: dict[str, object] | None,
    max_coarse_duration_seconds: float,
) -> str:
    previous_payload = (
        json.dumps(payload, ensure_ascii=False)
        if payload is not None
        else "null"
    )
    return (
        f"{base_prompt}\n\n"
        "LƯỢT SỬA BẮT BUỘC:\n"
        f"Phản hồi JSON trước bị validator từ chối: {error}\n"
        f"Phản hồi JSON trước: {previous_payload}\n"
        "Hãy trả lại TOÀN BỘ JSON đã sửa, không giải thích. Không lặp lại lỗi "
        "trên. Mỗi anchor phải là một chuỗi con chép nguyên văn từ đúng evidence, "
        "kể cả dấu câu và chính tả ASR. Mỗi block phải liên tục, tập trung vào "
        "roster trực tiếp và có độ dài thô không quá "
        f"{max_coarse_duration_seconds:g} giây. Nếu block trước quá dài hoặc đi "
        "qua nghi lễ/phân tích, hãy ưu tiên DỜI START tới câu tái giới thiệu "
        "lineup trực tiếp, dày đặc, muộn hơn. Không được cắt bỏ phần roster hoặc "
        "formation trực tiếp hợp lệ ở cuối chỉ để vượt validation. Trường reason "
        "chỉ được tóm tắt quyết định cuối trong tối đa hai câu; không ghi lại quá "
        "trình suy luận hoặc các phương án đã loại.\n"
    )


def _response_payload(response: object) -> dict[str, object]:
    try:
        message = qwen_message(response, operation="Qwen lineup detection")
    except QwenTranscriptionError as exc:
        raise LineupDetectionError(str(exc)) from exc
    raw_text = qwen_message_text(message)
    if not raw_text:
        raise LineupDetectionError("Qwen returned an empty lineup response.")
    if raw_text.startswith("```") and raw_text.endswith("```"):
        lines = raw_text.splitlines()
        if len(lines) >= 3 and lines[0].strip().casefold() in {"```", "```json"}:
            raw_text = "\n".join(lines[1:-1]).strip()
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


def _response_metrics(response: object) -> dict[str, object]:
    metrics: dict[str, object] = {}
    usage = response_to_plain(response_field(response, "usage"))
    if isinstance(usage, dict):
        metrics["usage"] = usage
    elapsed = response_field(response, "elapsed_seconds")
    if isinstance(elapsed, (int, float)):
        metrics["elapsed_seconds"] = round(float(elapsed), 3)
    request_id = response_field(response, "request_id")
    if request_id:
        metrics["request_id"] = str(request_id)
    return metrics


class QwenLineupDetector:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_TEXT_MODEL,
        base_url: str = DEFAULT_TEXT_BASE_URL,
        client: object | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
        expected_segment_count: int | None = None,
        max_coarse_duration_seconds: float = (
            DEFAULT_MAX_COARSE_LINEUP_DURATION_SECONDS
        ),
    ) -> None:
        if not model.strip():
            raise LineupDetectionError("Qwen text model cannot be empty.")
        if not base_url.strip():
            raise LineupDetectionError("Lineup model base URL cannot be empty.")
        if max_attempts <= 0 or retry_delay_seconds < 0:
            raise LineupDetectionError(
                "Qwen retry attempts must be positive and delay non-negative."
            )
        if expected_segment_count is not None and expected_segment_count <= 0:
            raise LineupDetectionError(
                "Expected lineup segment count must be positive."
            )
        if (
            not math.isfinite(max_coarse_duration_seconds)
            or max_coarse_duration_seconds <= 0
        ):
            raise LineupDetectionError(
                "Maximum coarse lineup duration must be positive."
            )
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.expected_segment_count = expected_segment_count
        self.max_coarse_duration_seconds = max_coarse_duration_seconds
        self.client = (
            client
            if client is not None
            else self._create_client(api_key, base_url)
        )

    @staticmethod
    def _create_client(api_key: str | None, base_url: str) -> object:
        return OpenAICompatibleLineupClient(base_url=base_url, api_key=api_key)

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
            max_coarse_duration_seconds=self.max_coarse_duration_seconds,
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
        attempt_prompt = prompt
        for attempt in range(1, self.max_attempts + 1):
            payload: dict[str, object] | None = None
            metrics: dict[str, object] = {}
            try:
                response = self._generate(attempt_prompt)
                metrics = _response_metrics(response)
                payload = _response_payload(response)
                result = parse_detection_result(
                    payload,
                    model=self.model,
                    known_evidence=evidence_bounds,
                    evidence_bounds=evidence_bounds,
                    evidence_texts=evidence_texts,
                    window_start=window_start_seconds,
                    window_end=window_end_seconds,
                    max_segment_duration_seconds=(
                        self.max_coarse_duration_seconds
                    ),
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
                    {
                        "attempt": attempt,
                        "status": "accepted",
                        **metrics,
                    }
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
                        **metrics,
                    }
                )
                if attempt >= self.max_attempts:
                    raise QwenLineupValidationError(
                        str(exc),
                        attempt_log,
                    ) from exc
                attempt_prompt = _repair_detection_prompt(
                    prompt,
                    error=exc,
                    payload=payload,
                    max_coarse_duration_seconds=(
                        self.max_coarse_duration_seconds
                    ),
                )
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
            "temperature": 0.0,
            "max_tokens": 2048,
            "response_format": {"type": "json_object"},
            "enable_thinking": False,
        }
        try:
            response = self.client.call(**call_kwargs)  # type: ignore[attr-defined]
            ensure_qwen_success(response, operation="Qwen lineup detection")
            return response
        except Exception as exc:
            raise QwenLineupRequestError(
                f"Qwen lineup detection request failed: {exc}"
            ) from exc
