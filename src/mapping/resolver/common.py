from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np
import pandas as pd


IGNORED_TEXT = {
    "bre", "coach", "crypto", "defence", "expedia", "fifaplus com", "football club",
    "forwards", "goalkeeper", "hisense", "hollywood", "lfc", "midfield",
    "premier league", "standard", "standard chartered", "starting formation",
    "substitutes", "team", "team formation", "formation", "vivo",
}
IGNORED_TEXT_MARKERS = {"crypto", "fifaplus", "hisense"}
TABLE_IGNORED_TEXT = IGNORED_TEXT | {
    "captain", "df", "elite", "fw", "gk", "head coach", "mf", "team lineup",
}


class LineupResolutionError(Exception): pass


@dataclass(slots=True)
class NumberCluster:
    observations: list[pd.Series] = field(default_factory=list)
    label_observations: list[tuple[str, float, int]] = field(default_factory=list)
    @property
    def center_x(self) -> float:
        return float(np.median([row["center_x_norm"] for row in self.observations]))
    @property
    def center_y(self) -> float:
        return float(np.median([row["center_y_norm"] for row in self.observations]))
    @property
    def frame_count(self) -> int:
        return len({int(row["frame_index"]) for row in self.observations})


@dataclass(slots=True)
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


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class SubstitutePanel:
    side: str
    boundary: float
    first_timestamp_seconds: float


def normalize_text(value: object) -> str:
    text = "".join(
        char for char in unicodedata.normalize("NFKD", str(value))
        if not unicodedata.combining(char)
    ).casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def similarity(left: object, right: object) -> float:
    left_normalized = normalize_text(left)
    right_normalized = normalize_text(right)
    if not left_normalized or not right_normalized:
        return 0.0
    return SequenceMatcher(None, left_normalized, right_normalized).ratio()


def is_name_like(text: object) -> bool:
    normalized = normalize_text(text)
    if not normalized or normalized in IGNORED_TEXT:
        return False
    if any(marker in normalized.replace(" ", "") for marker in IGNORED_TEXT_MARKERS):
        return False
    letters = sum(character.isalpha() for character in normalized)
    return letters >= 2 and len(normalized) <= 40


def is_table_name_like(text: object) -> bool:
    return normalize_text(text) not in TABLE_IGNORED_TEXT and is_name_like(text)


def parse_inline_player(text: object) -> tuple[int, str] | None:
    match = re.match(r"^\s*(\d{1,2})\s+(.+?)\s*$", str(text))
    if not match:
        return None
    shirt_number = int(match.group(1))
    player_name = match.group(2).strip()
    if not 1 <= shirt_number <= 99 or not is_table_name_like(player_name):
        return None
    return shirt_number, player_name


def substitute_headers(detections: pd.DataFrame) -> pd.DataFrame:
    if detections.empty:
        return detections
    is_header = detections["text"].map(normalize_text).str.contains(
        r"\bsubstitutes?\b", regex=True, na=False
    )
    return detections[is_header]


def find_substitute_panel(detections: pd.DataFrame) -> SubstitutePanel | None:
    headers = substitute_headers(detections)
    if headers.empty:
        return None
    left_headers, right_headers = (
        headers[headers["center_x_norm"] < 0.5],
        headers[headers["center_x_norm"] >= 0.5],
    )
    panel_headers = left_headers if len(left_headers) >= len(right_headers) else right_headers
    side = "left" if float(panel_headers["center_x_norm"].median()) < 0.5 else "right"
    header_x = float(panel_headers["center_x_norm"].median())
    boundary = min(0.5, header_x + 0.18) if side == "left" else max(0.5, header_x - 0.18)
    return SubstitutePanel(side, boundary, float(panel_headers["timestamp_seconds"].min()))


def detections_before_substitutes(segment: pd.DataFrame) -> pd.DataFrame:
    headers = substitute_headers(segment)
    if headers.empty:
        return segment
    cutoff = float(headers["timestamp_seconds"].min())
    return segment[segment["timestamp_seconds"] < cutoff]


def detections_without_substitute_panel(segment: pd.DataFrame) -> pd.DataFrame:
    panel = find_substitute_panel(segment)
    if panel is None:
        return segment
    panel_side = (segment["center_x_norm"] < panel.boundary if panel.side == "left"
                  else segment["center_x_norm"] > panel.boundary)
    earlier_panel = segment[
        panel_side
        & (segment["timestamp_seconds"] < panel.first_timestamp_seconds)
    ]
    starter_list_visible = any(
        sum(parse_inline_player(text) is not None for text in frame["text"]) >= 4
        for _, frame in earlier_panel.groupby("frame_index")
    )
    panel_frames = (
        pd.Series(True, index=segment.index)
        if starter_list_visible
        else segment["timestamp_seconds"] >= panel.first_timestamp_seconds
    )
    return segment[~(panel_frames & panel_side)]


def shirt_number_rows(group: pd.DataFrame) -> pd.DataFrame:
    numbers = group[group["text_type"] == "shirt_number_candidate"].copy()
    parsed = pd.to_numeric(numbers["text"], errors="coerce")
    return numbers[parsed.between(1, 99)].copy()


def spatial_distance(left: pd.Series, right: pd.Series) -> tuple[float, float]:
    return (
        abs(float(left["center_x_norm"]) - float(right["center_x_norm"])),
        abs(float(left["center_y_norm"]) - float(right["center_y_norm"])),
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
            if similarity(observation[0], group[0][0]) >= fuzzy_threshold:
                group.append(observation)
                break
        else:
            groups.append([observation])
    best_group = max(groups, key=lambda group: (
        len({frame for _, _, frame in group}), sum(score for _, score, _ in group),
    ))
    by_variant: defaultdict[str, list[tuple[float, int]]] = defaultdict(list)
    display_value: dict[str, str] = {}
    for text, score, frame in best_group:
        normalized = normalize_text(text)
        by_variant[normalized].append((score, frame))
        display_value.setdefault(normalized, text)
    best_variant = max(by_variant, key=lambda key: (
        len({frame for _, frame in by_variant[key]}),
        sum(score for score, _ in by_variant[key]),
    ))
    scores = [score for score, _ in by_variant[best_variant]]
    evidence_frames = {frame for _, _, frame in best_group}
    return display_value[best_variant], float(np.mean(scores)), len(evidence_frames)
