"""Gemini adapter and data contract for transcript-driven lineup detection."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from transcript.gemini import DEFAULT_MODEL
from transcript.schema import TranscriptSegment


DEFAULT_PROMPT_PATH = Path(__file__).parent / "prompts" / "detect_lineup_prompt.txt"
LINEUP_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "lineup_segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_seconds": {"type": "number", "minimum": 0},
                    "end_seconds": {"type": "number", "minimum": 0},
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "evidence_segment_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "reason": {"type": "string"},
                },
                "required": [
                    "start_seconds",
                    "end_seconds",
                    "confidence",
                    "evidence_segment_ids",
                    "reason",
                ],
            },
        }
    },
    "required": ["lineup_segments"],
}
CSV_COLUMNS = (
    "video",
    "segment_id",
    "start_seconds",
    "end_seconds",
    "confidence",
    "evidence_segment_ids",
    "reason",
    "model",
)


class LineupDetectionError(RuntimeError):
    """Raised when transcript-driven lineup detection is invalid or fails."""


@dataclass(frozen=True)
class LineupSegment:
    segment_id: str
    start_seconds: float
    end_seconds: float
    confidence: float
    evidence_segment_ids: tuple[str, ...]
    reason: str

    def validate(self) -> None:
        numeric_values = (self.start_seconds, self.end_seconds, self.confidence)
        if not all(math.isfinite(value) for value in numeric_values):
            raise LineupDetectionError("Lineup values must be finite numbers.")
        if self.start_seconds < 0 or self.end_seconds <= self.start_seconds:
            raise LineupDetectionError(
                f"Invalid lineup time range: {self.segment_id}"
            )
        if not 0 <= self.confidence <= 1:
            raise LineupDetectionError(
                f"Lineup confidence must be between 0 and 1: {self.segment_id}"
            )
        if not self.evidence_segment_ids:
            raise LineupDetectionError(
                f"Lineup evidence cannot be empty: {self.segment_id}"
            )
        if len(set(self.evidence_segment_ids)) != len(self.evidence_segment_ids):
            raise LineupDetectionError(
                f"Lineup evidence contains duplicate IDs: {self.segment_id}"
            )
        if not self.reason.strip():
            raise LineupDetectionError(
                f"Lineup reason cannot be empty: {self.segment_id}"
            )


@dataclass(frozen=True)
class LineupDetectionResult:
    model: str
    segments: tuple[LineupSegment, ...]
    raw_response: dict[str, Any]

    def validate(self) -> None:
        if not self.model.strip():
            raise LineupDetectionError("Detection model cannot be empty.")
        previous_start = -1.0
        seen_ids: set[str] = set()
        for segment in self.segments:
            segment.validate()
            if segment.segment_id in seen_ids:
                raise LineupDetectionError(
                    f"Duplicate lineup segment ID: {segment.segment_id}"
                )
            if segment.start_seconds < previous_start:
                raise LineupDetectionError(
                    "Lineup segments must be ordered by start_seconds."
                )
            previous_start = segment.start_seconds
            seen_ids.add(segment.segment_id)


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
    transcript_rows = [
        {
            "segment_id": segment.segment_id,
            "start_seconds": segment.start_seconds,
            "end_seconds": segment.end_seconds,
            "text": segment.text,
        }
        for segment in segments
    ]
    template = prompt_path.read_text(encoding="utf-8")
    replacements = {
        "{VIDEO_NAME}": video_name,
        "{WINDOW_START_SECONDS}": f"{window_start_seconds:.3f}",
        "{WINDOW_END_SECONDS}": f"{window_end_seconds:.3f}",
        "{TRANSCRIPT_JSON}": json.dumps(transcript_rows, ensure_ascii=False),
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template


class GeminiLineupDetector:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        client: object | None = None,
    ) -> None:
        if not model.strip():
            raise LineupDetectionError("Gemini model cannot be empty.")
        self.model = model
        if client is not None:
            self.client = client
            return
        if not api_key:
            raise LineupDetectionError(
                "Missing GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
            )
        try:
            from google import genai
        except ModuleNotFoundError as exc:
            raise LineupDetectionError(
                "google-genai is not installed. Install requirements.txt first."
            ) from exc
        self.client = genai.Client(api_key=api_key)

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
        if not video_name.strip():
            raise LineupDetectionError("video_name cannot be empty.")
        if (
            not math.isfinite(window_start_seconds)
            or not math.isfinite(window_end_seconds)
            or window_start_seconds < 0
            or window_end_seconds <= window_start_seconds
        ):
            raise LineupDetectionError("Invalid transcription window.")
        if not transcript:
            result = LineupDetectionResult(
                model=self.model,
                segments=tuple(),
                raw_response={"lineup_segments": []},
            )
            result.validate()
            return result

        prompt = build_detection_prompt(
            transcript,
            video_name=video_name,
            window_start_seconds=window_start_seconds,
            window_end_seconds=window_end_seconds,
            prompt_path=prompt_path,
        )
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": LINEUP_RESPONSE_SCHEMA,
                },
            )
            raw_text = getattr(response, "text", None)
            if not isinstance(raw_text, str) or not raw_text.strip():
                raise LineupDetectionError("Gemini returned an empty response.")
            try:
                payload = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                raise LineupDetectionError(
                    "Gemini lineup response is not valid JSON."
                ) from exc
            return self._parse_payload(
                payload,
                transcript=transcript,
                window_start_seconds=window_start_seconds,
                window_end_seconds=window_end_seconds,
            )
        except LineupDetectionError:
            raise
        except Exception as exc:
            raise LineupDetectionError(
                f"Gemini lineup detection request failed: {exc}"
            ) from exc

    def _parse_payload(
        self,
        payload: object,
        *,
        transcript: tuple[TranscriptSegment, ...],
        window_start_seconds: float,
        window_end_seconds: float,
    ) -> LineupDetectionResult:
        if not isinstance(payload, dict) or not isinstance(
            payload.get("lineup_segments"), list
        ):
            raise LineupDetectionError(
                "Gemini response must contain a lineup_segments array."
            )

        known_evidence = {segment.segment_id for segment in transcript}
        parsed: list[tuple[float, float, float, tuple[str, ...], str]] = []
        for index, item in enumerate(payload["lineup_segments"], start=1):
            if not isinstance(item, dict):
                raise LineupDetectionError(
                    f"Lineup candidate {index} must be an object."
                )
            try:
                start = float(item["start_seconds"])
                end = float(item["end_seconds"])
                confidence = float(item["confidence"])
            except (KeyError, TypeError, ValueError) as exc:
                raise LineupDetectionError(
                    f"Lineup candidate {index} has invalid numeric values."
                ) from exc
            raw_evidence = item.get("evidence_segment_ids")
            if not isinstance(raw_evidence, list) or not all(
                isinstance(value, str) and value.strip() for value in raw_evidence
            ):
                raise LineupDetectionError(
                    f"Lineup candidate {index} has invalid evidence IDs."
                )
            evidence = tuple(value.strip() for value in raw_evidence)
            unknown_evidence = sorted(set(evidence) - known_evidence)
            if unknown_evidence:
                raise LineupDetectionError(
                    f"Lineup candidate {index} cites unknown evidence: "
                    + ", ".join(unknown_evidence)
                )
            reason = str(item.get("reason", "")).strip()
            if start < window_start_seconds - 2 or end > window_end_seconds + 2:
                raise LineupDetectionError(
                    f"Lineup candidate {index} falls outside the processed window."
                )
            start = max(start, window_start_seconds)
            end = min(end, window_end_seconds)
            candidate = LineupSegment(
                segment_id="pending",
                start_seconds=start,
                end_seconds=end,
                confidence=confidence,
                evidence_segment_ids=evidence,
                reason=reason,
            )
            candidate.validate()
            parsed.append((start, end, confidence, evidence, reason))

        parsed.sort(key=lambda row: (row[0], row[1]))
        segments = tuple(
            LineupSegment(
                segment_id=f"lineup_{index:03d}",
                start_seconds=start,
                end_seconds=end,
                confidence=confidence,
                evidence_segment_ids=evidence,
                reason=reason,
            )
            for index, (start, end, confidence, evidence, reason) in enumerate(
                parsed, start=1
            )
        )
        result = LineupDetectionResult(
            model=self.model,
            segments=segments,
            raw_response=payload,
        )
        result.validate()
        return result


def write_lineup_csv(
    result: LineupDetectionResult,
    *,
    video_name: str,
    output_path: Path,
) -> None:
    result.validate()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for segment in result.segments:
            writer.writerow(
                {
                    "video": video_name,
                    "segment_id": segment.segment_id,
                    "start_seconds": round(segment.start_seconds, 3),
                    "end_seconds": round(segment.end_seconds, 3),
                    "confidence": round(segment.confidence, 4),
                    "evidence_segment_ids": json.dumps(
                        segment.evidence_segment_ids, ensure_ascii=False
                    ),
                    "reason": segment.reason,
                    "model": result.model,
                }
            )
