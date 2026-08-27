"""Validated data contract for detected lineup intervals."""

from __future__ import annotations

import csv
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping


LINEUP_FIELDS = {
    "team_name": {"type": "string", "minLength": 1},
    "start_seconds": {"type": "number", "minimum": 0},
    "end_seconds": {"type": "number", "minimum": 0},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "context": {
        "type": "string",
        "enum": [
            "lineup_presentation",
            "player_walkout",
            "officials_or_ceremony",
            "live_play_tactical_analysis",
            "uncertain",
        ],
    },
    "evidence_segment_ids": {
        "type": "array",
        "items": {"type": "string"},
    },
    "reason": {"type": "string"},
    "start_anchor_text": {"type": "string"},
    "end_anchor_text": {"type": "string"},
}
REQUIRED_LINEUP_FIELDS = (
    "team_name",
    "start_seconds",
    "end_seconds",
    "confidence",
    "context",
    "evidence_segment_ids",
    "reason",
    "start_anchor_text",
    "end_anchor_text",
)
LINEUP_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "lineup_segments": {
            "type": "array",
            "maxItems": 2,
            "items": {
                "type": "object",
                "properties": LINEUP_FIELDS,
                "required": list(REQUIRED_LINEUP_FIELDS),
            },
        },
    },
    "required": ["lineup_segments"],
}
CSV_COLUMNS = (
    "video segment_id detection_status team_name start_seconds end_seconds confidence "
    "evidence_start_seconds evidence_end_seconds evidence_segment_ids "
    "start_anchor_text end_anchor_text reason model"
).split()
MAX_EVIDENCE_GAP_SECONDS = 30.0
MAX_LINEUP_SEGMENTS = 2
DETECTION_STATUSES = {"complete", "incomplete", "empty", "unconstrained"}


class LineupDetectionError(RuntimeError):
    """Raised when transcript-driven lineup detection is invalid or fails."""


class RejectedLineupCandidate(LineupDetectionError):
    """Raised for one unsafe candidate while preserving other valid rows."""


@dataclass(frozen=True)
class LineupSegment:
    segment_id: str
    team_name: str
    start_seconds: float
    end_seconds: float
    confidence: float
    evidence_segment_ids: tuple[str, ...]
    start_anchor_text: str
    end_anchor_text: str
    reason: str
    evidence_start_seconds: float | None = None
    evidence_end_seconds: float | None = None

    def validate(self) -> None:
        values = (self.start_seconds, self.end_seconds, self.confidence)
        if not all(math.isfinite(value) for value in values):
            raise LineupDetectionError("Lineup values must be finite numbers.")
        if self.start_seconds < 0 or self.end_seconds <= self.start_seconds:
            raise LineupDetectionError(f"Invalid lineup range: {self.segment_id}")
        if not 0 <= self.confidence <= 1:
            raise LineupDetectionError(
                f"Lineup confidence must be between 0 and 1: {self.segment_id}"
            )
        if not self.evidence_segment_ids:
            raise LineupDetectionError(f"Missing evidence: {self.segment_id}")
        if len(set(self.evidence_segment_ids)) != len(self.evidence_segment_ids):
            raise LineupDetectionError(
                f"Duplicate evidence IDs: {self.segment_id}"
            )
        if not self.team_name.strip():
            raise LineupDetectionError(f"Missing team name: {self.segment_id}")
        if not self.start_anchor_text.strip() or not self.end_anchor_text.strip():
            raise LineupDetectionError(f"Missing text anchors: {self.segment_id}")
        if not self.reason.strip():
            raise LineupDetectionError(f"Missing reason: {self.segment_id}")
        evidence_values = (
            self.evidence_start_seconds,
            self.evidence_end_seconds,
        )
        if any(
            value is not None and not math.isfinite(value)
            for value in evidence_values
        ):
            raise LineupDetectionError(
                f"Evidence bounds must be finite: {self.segment_id}"
            )
        if (
            self.evidence_start_seconds is not None
            and self.evidence_end_seconds is not None
            and (
                self.evidence_start_seconds < 0
                or self.evidence_end_seconds <= self.evidence_start_seconds
            )
        ):
            raise LineupDetectionError(
                f"Invalid evidence bounds: {self.segment_id}"
            )

    def csv_row(self, *, video_name: str, model: str) -> dict[str, object]:
        row = asdict(self)
        row["video"], row["model"] = video_name, model
        row["start_seconds"] = round(self.start_seconds, 3)
        row["end_seconds"] = round(self.end_seconds, 3)
        row["confidence"] = round(self.confidence, 4)
        for field in ("evidence_start_seconds", "evidence_end_seconds"):
            value = row[field]
            row[field] = "" if value is None else round(float(value), 3)
        row["evidence_segment_ids"] = json.dumps(
            self.evidence_segment_ids, ensure_ascii=False
        )
        return row


@dataclass(frozen=True)
class LineupDetectionResult:
    model: str
    segments: tuple[LineupSegment, ...]
    raw_response: dict[str, Any]
    status: str = "unconstrained"

    def validate(self) -> None:
        if not self.model.strip():
            raise LineupDetectionError("Detection model cannot be empty.")
        if self.status not in DETECTION_STATUSES:
            raise LineupDetectionError(f"Invalid detection status: {self.status}")
        if len(self.segments) > MAX_LINEUP_SEGMENTS:
            raise LineupDetectionError(
                f"At most {MAX_LINEUP_SEGMENTS} lineup segments are supported."
            )
        seen: set[str] = set()
        seen_teams: set[str] = set()
        previous_start = -1.0
        previous_end = -1.0
        for segment in self.segments:
            segment.validate()
            if segment.segment_id in seen:
                raise LineupDetectionError(
                    f"Duplicate lineup segment ID: {segment.segment_id}"
                )
            if segment.start_seconds < previous_start:
                raise LineupDetectionError(
                    "Lineup segments must be ordered by start_seconds."
                )
            team_key = unicodedata.normalize(
                "NFKC", segment.team_name
            ).strip().casefold()
            if team_key in seen_teams:
                raise LineupDetectionError(
                    f"Lineup segments must belong to distinct teams: {segment.team_name}"
                )
            if segment.start_seconds < previous_end - 1e-6:
                raise LineupDetectionError("Lineup segments must not overlap.")
            seen.add(segment.segment_id)
            seen_teams.add(team_key)
            previous_start = segment.start_seconds
            previous_end = segment.end_seconds


def _normalized_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _parse_candidate(
    item: object,
    *,
    index: int,
    known_evidence: set[str],
    evidence_bounds: Mapping[str, tuple[float, float]] | None,
    evidence_texts: Mapping[str, str] | None,
    window_start: float,
    window_end: float,
) -> LineupSegment | None:
    if not isinstance(item, dict):
        raise LineupDetectionError(f"Candidate {index} must be an object.")

    context = str(item.get("context", "")).strip()
    allowed_contexts = {
        "lineup_presentation",
        "player_walkout",
        "officials_or_ceremony",
        "live_play_tactical_analysis",
        "uncertain",
    }
    if context not in allowed_contexts:
        raise LineupDetectionError(
            f"Candidate {index} has invalid presentation context."
        )
    if context != "lineup_presentation":
        return None

    try:
        confidence = float(item["confidence"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LineupDetectionError(
            f"Candidate {index} has invalid confidence."
        ) from exc
    try:
        start = float(item["start_seconds"])
        end = float(item["end_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LineupDetectionError(
            f"Candidate {index} has invalid timestamps."
        ) from exc

    team_name = str(item.get("team_name", "")).strip()
    reason = str(item.get("reason", "")).strip()
    start_anchor = str(item.get("start_anchor_text", "")).strip()
    end_anchor = str(item.get("end_anchor_text", "")).strip()
    if not team_name:
        raise LineupDetectionError(f"Candidate {index} has no team_name.")
    if not reason:
        raise LineupDetectionError(f"Candidate {index} has no reason.")
    if not start_anchor or not end_anchor:
        raise LineupDetectionError(
            f"Candidate {index} must contain both exact text anchors."
        )

    raw_evidence = item.get("evidence_segment_ids")
    evidence = (
        tuple(value.strip() for value in raw_evidence)
        if isinstance(raw_evidence, list)
        and all(isinstance(value, str) and value.strip() for value in raw_evidence)
        else ()
    )
    unknown = set(evidence) - known_evidence
    if not evidence or unknown:
        raise LineupDetectionError(f"Candidate {index} has invalid evidence IDs.")
    evidence_start: float | None = None
    evidence_end: float | None = None
    if evidence_bounds is not None and evidence:
        cited_bounds = [evidence_bounds[evidence_id] for evidence_id in evidence]
        evidence_start = min(bound[0] for bound in cited_bounds)
        evidence_end = max(bound[1] for bound in cited_bounds)

    def anchor_time(anchor: str, field: str, *, use_end: bool) -> float | None:
        if evidence_bounds is None or evidence_texts is None:
            return None
        normalized_anchor = _normalized_text(anchor)
        matches: list[tuple[str, int, int]] = []
        for evidence_id in evidence:
            text = evidence_texts[evidence_id]
            normalized_evidence = _normalized_text(text)
            position = normalized_evidence.find(normalized_anchor)
            while position >= 0:
                matches.append((evidence_id, position, len(normalized_evidence)))
                position = normalized_evidence.find(
                    normalized_anchor, position + max(1, len(normalized_anchor))
                )
        if len(matches) != 1:
            detail = "not found" if not matches else "ambiguous"
            raise LineupDetectionError(
                f"Candidate {index} {field} is {detail} in cited evidence."
            )
        evidence_id, position, text_length = matches[0]
        if use_end:
            position += len(normalized_anchor)
        segment_start, segment_end = evidence_bounds[evidence_id]
        fraction = position / max(1, text_length)
        return segment_start + (segment_end - segment_start) * fraction

    anchored_start = anchor_time(
        start_anchor, "start_anchor_text", use_end=False
    )
    anchored_end = anchor_time(end_anchor, "end_anchor_text", use_end=True)
    if evidence_bounds is not None and evidence_texts is not None and (
        anchored_start is None or anchored_end is None
    ):
        raise LineupDetectionError(
            f"Candidate {index} could not align both text anchors."
        )
    if anchored_start is not None:
        start = anchored_start
    if anchored_end is not None:
        end = anchored_end
    if (
        evidence_start is not None
        and evidence_end is not None
        and (start < evidence_start - 2 or end > evidence_end + 2)
    ):
        raise LineupDetectionError(
            f"Candidate {index} timestamps fall outside cited evidence."
        )
    if start < window_start - 2 or end > window_end + 2:
        raise LineupDetectionError(f"Candidate {index} is outside the window.")

    ceremony_markers = (
        "introducing the players one by one",
        "introduced one by one",
        "next out",
        "will be next out",
        "will follow",
        "leads out",
        "leads us out",
        "last out",
        "make their way out",
    )
    ceremony_parts = [
        start_anchor,
        end_anchor,
    ]
    ceremony_context = _normalized_text(" ".join(ceremony_parts))
    if any(
        re.search(rf"\b{re.escape(marker)}\b", ceremony_context)
        for marker in ceremony_markers
    ):
        raise RejectedLineupCandidate(
            f"Candidate {index} anchors contain player-walkout language."
        )

    if evidence_texts is not None:
        ordered_evidence = (
            sorted(evidence, key=lambda evidence_id: evidence_bounds[evidence_id])
            if evidence_bounds is not None
            else list(evidence)
        )
        evidence_block = _normalized_text(
            " ".join(evidence_texts[evidence_id] for evidence_id in ordered_evidence)
        )
        normalized_start = _normalized_text(start_anchor)
        normalized_end = _normalized_text(end_anchor)
        block_start = evidence_block.find(normalized_start)
        block_end = evidence_block.rfind(normalized_end)
        if block_start >= 0 and block_end >= block_start:
            block_end += len(normalized_end)
            anchored_block = evidence_block[block_start:block_end]
            if any(
                re.search(rf"\b{re.escape(marker)}\b", anchored_block)
                for marker in ceremony_markers
            ):
                raise RejectedLineupCandidate(
                    f"Candidate {index} is part of a player-walkout sequence."
                )

    segment = LineupSegment(
        segment_id="pending",
        team_name=team_name,
        start_seconds=max(start, window_start),
        end_seconds=min(end, window_end),
        confidence=confidence,
        evidence_segment_ids=evidence,
        start_anchor_text=start_anchor,
        end_anchor_text=end_anchor,
        reason=reason,
        evidence_start_seconds=evidence_start,
        evidence_end_seconds=evidence_end,
    )
    segment.validate()
    return segment


def _split_candidate_by_evidence_gaps(
    segment: LineupSegment,
    *,
    evidence_bounds: Mapping[str, tuple[float, float]] | None,
    max_gap_seconds: float = MAX_EVIDENCE_GAP_SECONDS,
) -> tuple[LineupSegment, ...]:
    """Split a model candidate when its cited evidence is not continuous."""
    if evidence_bounds is None or len(segment.evidence_segment_ids) < 2:
        return (segment,)
    ordered = sorted(
        segment.evidence_segment_ids,
        key=lambda evidence_id: evidence_bounds[evidence_id],
    )
    groups: list[list[str]] = [[ordered[0]]]
    previous_end = evidence_bounds[ordered[0]][1]
    for evidence_id in ordered[1:]:
        start, end = evidence_bounds[evidence_id]
        if start - previous_end > max_gap_seconds:
            groups.append([])
        groups[-1].append(evidence_id)
        previous_end = max(previous_end, end)
    if len(groups) == 1:
        return (segment,)

    split: list[LineupSegment] = []
    for group in groups:
        group_start = min(evidence_bounds[evidence_id][0] for evidence_id in group)
        group_end = max(evidence_bounds[evidence_id][1] for evidence_id in group)
        start = max(segment.start_seconds, group_start)
        end = min(segment.end_seconds, group_end)
        if end <= start:
            continue
        piece = replace(
            segment,
            start_seconds=start,
            end_seconds=end,
            evidence_segment_ids=tuple(group),
            reason=(
                segment.reason
                + " Evidence was split at a non-lineup temporal gap."
            ),
        )
        piece.validate()
        split.append(piece)
    return tuple(split) or (segment,)


def parse_detection_result(
    payload: object,
    *,
    model: str,
    known_evidence: Iterable[str],
    evidence_bounds: Mapping[str, tuple[float, float]] | None = None,
    evidence_texts: Mapping[str, str] | None = None,
    window_start: float,
    window_end: float,
) -> LineupDetectionResult:
    if not isinstance(payload, dict) or not isinstance(
        payload.get("lineup_segments"), list
    ):
        raise LineupDetectionError(
            "Model response must contain a lineup_segments array."
        )
    evidence_ids = set(known_evidence)
    candidates: list[LineupSegment | None] = []
    rejected_candidates: list[dict[str, object]] = []
    for index, item in enumerate(payload["lineup_segments"], start=1):
        try:
            candidates.append(
                _parse_candidate(
                    item,
                    index=index,
                    known_evidence=evidence_ids,
                    evidence_bounds=evidence_bounds,
                    evidence_texts=evidence_texts,
                    window_start=window_start,
                    window_end=window_end,
                )
            )
        except RejectedLineupCandidate as exc:
            rejected_candidates.append(
                {"candidate_index": index, "reason": str(exc)}
            )
    parsed = sorted(
        (
            piece
            for candidate in candidates
            if candidate is not None
            for piece in _split_candidate_by_evidence_gaps(
                candidate,
                evidence_bounds=evidence_bounds,
            )
        ),
        key=lambda segment: (segment.start_seconds, segment.end_seconds),
    )
    segments = tuple(
        replace(segment, segment_id=f"lineup_{index:03d}")
        for index, segment in enumerate(parsed, start=1)
    )
    raw_response = dict(payload)
    if rejected_candidates:
        raw_response["_rejected_candidates"] = rejected_candidates
    result = LineupDetectionResult(model, segments, raw_response)
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
        rows = []
        for segment in result.segments:
            row = segment.csv_row(video_name=video_name, model=result.model)
            row["detection_status"] = result.status
            rows.append(row)
        writer.writerows(rows)
