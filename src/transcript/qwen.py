"""DashScope Qwen adapter for chunked speech recognition."""

from __future__ import annotations

import math
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any, Callable, TypeVar

from .schema import TranscriptSegment, TranscriptionResult


DEFAULT_ASR_MODEL = "qwen3-asr-flash"
DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/api/v1"
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_DELAY_SECONDS = 2.0
TRANSIENT_ERROR_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "connection reset",
    "deadline exceeded",
    "internal",
    "rate limit",
    "temporarily unavailable",
    "timeout",
    "too many requests",
    "unavailable",
)
ResultT = TypeVar("ResultT")


class QwenTranscriptionError(RuntimeError):
    """Raised when Qwen cannot produce a usable transcription."""


def is_transient_qwen_error(error: BaseException) -> bool:
    """Return whether retrying a DashScope request is likely to help."""
    message = str(error).lower()
    return any(marker in message for marker in TRANSIENT_ERROR_MARKERS)


def call_with_qwen_retry(
    operation: Callable[[], ResultT],
    *,
    max_attempts: int,
    retry_delay_seconds: float,
    label: str,
) -> ResultT:
    """Retry one transient Qwen operation with bounded exponential backoff."""
    if max_attempts <= 0 or retry_delay_seconds < 0:
        raise ValueError("Invalid Qwen retry configuration.")
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt >= max_attempts or not is_transient_qwen_error(exc):
                raise
            delay = retry_delay_seconds * (2 ** (attempt - 1))
            print(
                f"Transient {label} error; retrying in {delay:g}s "
                f"({attempt + 1}/{max_attempts})."
            )
            time.sleep(delay)
    raise RuntimeError(f"{label} failed without an error.")


def response_field(value: object, name: str, default: Any = None) -> Any:
    """Read one field from either a DashScope object or a plain mapping."""
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def response_to_plain(value: object) -> Any:
    """Convert the small DashScope response fragments we persist to JSON values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): response_to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [response_to_plain(item) for item in value]
    if hasattr(value, "items"):
        try:
            return {
                str(key): response_to_plain(item)
                for key, item in value.items()  # type: ignore[attr-defined]
            }
        except (AttributeError, TypeError, ValueError):
            pass
    return str(value)


def ensure_qwen_success(response: object, *, operation: str) -> None:
    """Raise a readable error for a non-2xx DashScope response."""
    status = response_field(response, "status_code")
    if status is None:
        return
    try:
        status_code = int(status)
    except (TypeError, ValueError):
        status_code = status
    if status_code == int(HTTPStatus.OK):
        return
    code = response_field(response, "code", "unknown_error")
    message = response_field(response, "message", "No error message")
    request_id = response_field(response, "request_id", "unknown")
    raise QwenTranscriptionError(
        f"{operation} failed ({status_code}, {code}, request {request_id}): {message}"
    )


def qwen_message(response: object, *, operation: str) -> object:
    """Return output.choices[0].message from a successful DashScope response."""
    ensure_qwen_success(response, operation=operation)
    output = response_field(response, "output")
    choices = response_field(output, "choices")
    if not isinstance(choices, (list, tuple)) or not choices:
        raise QwenTranscriptionError(f"{operation} returned no choices.")
    message = response_field(choices[0], "message")
    if message is None:
        raise QwenTranscriptionError(f"{operation} returned no message.")
    return message


def qwen_message_text(message: object) -> str:
    """Extract text from either multimodal or text-generation message content."""
    content = response_field(message, "content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, (list, tuple)):
        return ""
    text_parts: list[str] = []
    for item in content:
        text = response_field(item, "text")
        if isinstance(text, str) and text.strip():
            text_parts.append(text.strip())
    return "\n".join(text_parts).strip()


def configure_dashscope(base_url: str) -> object:
    """Configure the SDK endpoint and return its multimodal API class."""
    try:
        import dashscope
    except ModuleNotFoundError as exc:
        raise QwenTranscriptionError(
            "dashscope is not installed. Install requirements.txt first."
        ) from exc
    dashscope.base_http_api_url = base_url
    return dashscope.MultiModalConversation


class QwenTranscriber:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_ASR_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        client: object | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
    ) -> None:
        if not model.strip():
            raise QwenTranscriptionError("Qwen ASR model cannot be empty.")
        if not base_url.strip():
            raise QwenTranscriptionError("DashScope base URL cannot be empty.")
        if max_attempts <= 0 or retry_delay_seconds < 0:
            raise QwenTranscriptionError(
                "Qwen retry attempts must be positive and delay non-negative."
            )
        if client is None and not api_key:
            raise QwenTranscriptionError(
                "Missing DASHSCOPE_API_KEY environment variable."
            )
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.max_attempts = max_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.client = client if client is not None else configure_dashscope(base_url)

    def transcribe(
        self,
        audio_path: Path,
        *,
        audio_duration_seconds: float,
        language: str = "auto",
        source_chunk: str = "chunk_000",
        chunk_start_seconds: float = 0.0,
        context: str | None = None,
        raw_response_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> TranscriptionResult:
        audio = audio_path.expanduser().resolve()
        if not audio.is_file():
            raise QwenTranscriptionError(f"Audio file does not exist: {audio}")
        if not math.isfinite(audio_duration_seconds) or audio_duration_seconds <= 0:
            raise QwenTranscriptionError("audio_duration_seconds must be positive.")
        if not math.isfinite(chunk_start_seconds) or chunk_start_seconds < 0:
            raise QwenTranscriptionError("chunk_start_seconds must be non-negative.")
        if not source_chunk.strip():
            raise QwenTranscriptionError("source_chunk cannot be empty.")

        return call_with_qwen_retry(
            lambda: self._transcribe_once(
                audio,
                audio_duration_seconds=audio_duration_seconds,
                language=language,
                source_chunk=source_chunk,
                chunk_start_seconds=chunk_start_seconds,
                context=context,
                raw_response_callback=raw_response_callback,
            ),
            max_attempts=self.max_attempts,
            retry_delay_seconds=self.retry_delay_seconds,
            label="Qwen ASR",
        )

    def _transcribe_once(
        self,
        audio: Path,
        *,
        audio_duration_seconds: float,
        language: str,
        source_chunk: str,
        chunk_start_seconds: float,
        context: str | None,
        raw_response_callback: Callable[[dict[str, Any]], None] | None,
    ) -> TranscriptionResult:
        messages: list[dict[str, object]] = []
        if context and context.strip():
            messages.append(
                {"role": "system", "content": [{"text": context.strip()}]}
            )
        messages.append(
            {"role": "user", "content": [{"audio": str(audio)}]}
        )
        asr_options: dict[str, object] = {"enable_itn": True}
        normalized_language = language.strip().lower()
        if normalized_language and normalized_language != "auto":
            asr_options["language"] = normalized_language

        call_kwargs: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "result_format": "message",
            "asr_options": asr_options,
        }
        if self.api_key:
            call_kwargs["api_key"] = self.api_key
        try:
            response = self.client.call(**call_kwargs)  # type: ignore[attr-defined]
            message = qwen_message(response, operation="Qwen ASR")
            text = qwen_message_text(message)
            annotations = response_field(message, "annotations", [])
            detected_language = None
            if isinstance(annotations, (list, tuple)):
                for annotation in annotations:
                    candidate = response_field(annotation, "language")
                    if isinstance(candidate, str) and candidate.strip():
                        detected_language = candidate.strip()
                        break
            raw_segments = []
            if text:
                raw_segments.append(
                    {
                        "start_seconds": 0.0,
                        "end_seconds": round(audio_duration_seconds, 3),
                        "text": text,
                    }
                )
            payload: dict[str, Any] = {
                "request_id": response_field(response, "request_id"),
                "detected_language": detected_language,
                "annotations": response_to_plain(annotations),
                "usage": response_to_plain(response_field(response, "usage", {})),
                "segments": raw_segments,
            }
            if raw_response_callback is not None:
                raw_response_callback(payload)
        except QwenTranscriptionError:
            raise
        except Exception as exc:
            raise QwenTranscriptionError(f"Qwen ASR request failed: {exc}") from exc

        segments: tuple[TranscriptSegment, ...] = ()
        if text:
            segments = (
                TranscriptSegment(
                    segment_id=f"{source_chunk}_seg_0001",
                    start_seconds=chunk_start_seconds,
                    end_seconds=chunk_start_seconds + audio_duration_seconds,
                    text=text,
                    source_chunk=source_chunk,
                    chunk_start_seconds=chunk_start_seconds,
                ),
            )
        result = TranscriptionResult(
            model=self.model,
            segments=segments,
            raw_response=payload,
        )
        result.validate()
        return result
