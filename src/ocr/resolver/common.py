"""Shared types, text normalization, validation, and consensus helpers."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REQUIRED_COLUMNS = {
    "video",
    "segment_index",
    "segment_start_seconds",
    "segment_end_seconds",
    "frame_index",
    "timestamp_seconds",
    "text",
    "text_type",
    "score",
    "center_x_norm",
    "center_y_norm",
}
IGNORED_TEXT = {
    "bre",
    "crypto",
    "defence",
    "expedia",
    "fifaplus com",
    "football club",
    "forwards",
    "goalkeeper",
    "hisense",
    "hollywood",
    "lfc",
    "midfield",
    "premier league",
    "standard",
    "standard chartered",
    "starting formation",
    "substitutes",
    "team",
    "team formation",
    "formation",
    "vivo",
}
IGNORED_TEXT_MARKERS = {
    "crypto",
    "fifaplus",
    "hisense",
}
TABLE_IGNORED_TEXT = IGNORED_TEXT | {
    "captain",
    "coach",
    "df",
    "elite",
    "fw",
    "gk",
    "head coach",
    "mf",
    "team lineup",
}


class LineupResolutionError(Exception):
    """Raised when raw OCR detections cannot be resolved into a lineup."""


@dataclass
class NumberCluster:
    observations: list[pd.Series] = field(default_factory=list)
    label_observations: list[tuple[str, float, int]] = field(
        default_factory=list
    )

    @property
    def center_x(self) -> float:
        return float(
            np.median(
                [
                    float(row["center_x_norm"])
                    for row in self.observations
                ]
            )
        )

    @property
    def center_y(self) -> float:
        return float(
            np.median(
                [
                    float(row["center_y_norm"])
                    for row in self.observations
                ]
            )
        )

    @property
    def frame_count(self) -> int:
        return len(
            {int(row["frame_index"]) for row in self.observations}
        )


@dataclass
class FormationEvent:
    snapshot_frames: list[int]
    snapshot_timestamps: list[float]
    signatures: list[list[pd.Series]]

    @property
    def start_seconds(self) -> float:
        return min(self.snapshot_timestamps)

    @property
    def end_seconds(self) -> float:
        return max(self.snapshot_timestamps)


@dataclass(frozen=True)
class PairObservation:
    shirt_number: int
    player_name: str
    number_score: float
    name_score: float
    frame_index: int
    timestamp_seconds: float
    center_x: float
    center_y: float
    label_x1: float
    relation: str


def resolve_project_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def similarity(left: object, right: object) -> float:
    left_normalized = normalize_text(left)
    right_normalized = normalize_text(right)
    if not left_normalized or not right_normalized:
        return 0.0
    return SequenceMatcher(
        None,
        left_normalized,
        right_normalized,
    ).ratio()


def is_name_like(text: object) -> bool:
    normalized = normalize_text(text)
    if not normalized or normalized in IGNORED_TEXT:
        return False
    compact = normalized.replace(" ", "")
    if any(marker in compact for marker in IGNORED_TEXT_MARKERS):
        return False
    letters = sum(character.isalpha() for character in normalized)
    if letters < 2 or len(normalized) > 40:
        return False
    return not normalized.isdigit()


def is_table_name_like(text: object) -> bool:
    normalized = normalize_text(text)
    if normalized in TABLE_IGNORED_TEXT:
        return False
    return is_name_like(text)


def parse_inline_player(text: object) -> tuple[int, str] | None:
    match = re.match(r"^\s*(\d{1,2})\s+(.+?)\s*$", str(text))
    if not match:
        return None
    shirt_number = int(match.group(1))
    player_name = match.group(2).strip()
    if not 1 <= shirt_number <= 99 or not is_table_name_like(player_name):
        return None
    return shirt_number, player_name


def detections_before_substitutes(
    segment: pd.DataFrame,
) -> pd.DataFrame:
    substitute_rows = segment[
        segment["text"].map(normalize_text).str.contains(
            r"\bsubstitutes?\b",
            regex=True,
            na=False,
        )
    ]
    if substitute_rows.empty:
        return segment
    cutoff = float(substitute_rows["timestamp_seconds"].min())
    return segment[segment["timestamp_seconds"] < cutoff]


def detections_without_substitute_panel(
    segment: pd.DataFrame,
) -> pd.DataFrame:
    headers = segment[
        segment["text"].map(normalize_text).str.contains(
            r"\bsubstitutes?\b",
            regex=True,
            na=False,
        )
    ]
    if headers.empty:
        return segment

    left_headers = headers[headers["center_x_norm"] < 0.5]
    right_headers = headers[headers["center_x_norm"] >= 0.5]
    panel_headers = (
        left_headers
        if len(left_headers) >= len(right_headers)
        else right_headers
    )
    panel_is_left = float(panel_headers["center_x_norm"].median()) < 0.5
    header_x = float(panel_headers["center_x_norm"].median())
    first_panel_timestamp = float(
        panel_headers["timestamp_seconds"].min()
    )
    panel_boundary = (
        min(0.5, header_x + 0.18)
        if panel_is_left
        else max(0.5, header_x - 0.18)
    )

    panel_frames = segment["timestamp_seconds"] >= first_panel_timestamp
    panel_side = (
        segment["center_x_norm"] < panel_boundary
        if panel_is_left
        else segment["center_x_norm"] > panel_boundary
    )
    return segment[~(panel_frames & panel_side)]


def load_detections(input_csv: Path) -> pd.DataFrame:
    if not input_csv.is_file():
        raise LineupResolutionError(f"OCR CSV does not exist: {input_csv}")

    detections = pd.read_csv(input_csv)
    missing = sorted(REQUIRED_COLUMNS - set(detections.columns))
    if missing:
        raise LineupResolutionError(
            "OCR CSV is missing required columns: " + ", ".join(missing)
        )
    if detections.empty:
        raise LineupResolutionError(f"OCR CSV has no rows: {input_csv}")

    numeric_columns = [
        "segment_index",
        "segment_start_seconds",
        "segment_end_seconds",
        "frame_index",
        "timestamp_seconds",
        "score",
        "center_x_norm",
        "center_y_norm",
    ]
    for column in numeric_columns:
        detections[column] = pd.to_numeric(
            detections[column],
            errors="raise",
        )

    values = detections[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise LineupResolutionError(
            "OCR CSV contains non-finite numeric values."
        )
    if not detections["score"].between(0, 1).all():
        raise LineupResolutionError(
            "OCR confidence must be between 0 and 1."
        )
    if not detections["center_x_norm"].between(0, 1).all():
        raise LineupResolutionError(
            "center_x_norm must be between 0 and 1."
        )
    if not detections["center_y_norm"].between(0, 1).all():
        raise LineupResolutionError(
            "center_y_norm must be between 0 and 1."
        )

    detections["text"] = detections["text"].astype(str).str.strip()
    return detections[detections["text"] != ""].copy()


def shirt_number_rows(group: pd.DataFrame) -> pd.DataFrame:
    numbers = group[
        group["text_type"] == "shirt_number_candidate"
    ].copy()
    parsed = pd.to_numeric(numbers["text"], errors="coerce")
    return numbers[parsed.between(1, 99)].copy()


def spatial_distance(
    left: pd.Series,
    right: pd.Series,
) -> tuple[float, float]:
    return (
        abs(
            float(left["center_x_norm"])
            - float(right["center_x_norm"])
        ),
        abs(
            float(left["center_y_norm"])
            - float(right["center_y_norm"])
        ),
    )


def consensus_text(
    observations: list[tuple[str, float, int]],
    fuzzy_threshold: float = 0.84,
) -> tuple[str, float, int]:
    if not observations:
        return "", 0.0, 0

    groups: list[list[tuple[str, float, int]]] = []
    for observation in observations:
        for group in groups:
            if similarity(
                observation[0],
                group[0][0],
            ) >= fuzzy_threshold:
                group.append(observation)
                break
        else:
            groups.append([observation])

    best_group = max(
        groups,
        key=lambda group: (
            len({frame for _, _, frame in group}),
            sum(score for _, score, _ in group),
        ),
    )
    by_variant: defaultdict[str, list[tuple[float, int]]] = defaultdict(
        list
    )
    display_value: dict[str, str] = {}
    for text, score, frame in best_group:
        normalized = normalize_text(text)
        by_variant[normalized].append((score, frame))
        display_value.setdefault(normalized, text)

    best_variant = max(
        by_variant,
        key=lambda key: (
            len({frame for _, frame in by_variant[key]}),
            sum(score for score, _ in by_variant[key]),
        ),
    )
    scores = [score for score, _ in by_variant[best_variant]]
    evidence_frames = {frame for _, _, frame in best_group}
    return (
        display_value[best_variant],
        float(np.mean(scores)),
        len(evidence_frames),
    )
