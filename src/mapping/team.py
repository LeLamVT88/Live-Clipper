from __future__ import annotations

from difflib import SequenceMatcher
import logging
import re
import unicodedata
from typing import Callable, Protocol

import numpy as np

from mapping.config import MappingConfig
from mapping.regions import crop_region
from mapping.schema import FrameSample, TeamBoundary, TeamObservation
from mapping.video import extract_frame


LOGGER = logging.getLogger(__name__)


class TextReader(Protocol):
    def read_text(self, image: np.ndarray, *, digits_only: bool = False) -> tuple[str, float]: ...


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(character for character in value if not unicodedata.combining(character))
    return re.sub(r"[^A-Z0-9]+", " ", value.upper()).strip()


def similarity(left: str, right: str) -> float:
    try:
        from rapidfuzz.fuzz import WRatio
        return float(WRatio(normalize_text(left), normalize_text(right)))
    except ImportError:
        return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio() * 100


def match_team_name(raw_text: str, teams: list[str], threshold: float) -> tuple[str | None, float]:
    if not raw_text.strip() or not teams:
        return None, 0.0
    ranked = sorted(((similarity(raw_text, team), team) for team in teams), reverse=True)
    score, team = ranked[0]
    return (team if score >= threshold else None), score


def read_team_observations(
    frames: list[FrameSample], teams: list[str], config: MappingConfig, reader: TextReader,
) -> list[TeamObservation]:
    crops = [crop_region(sample.image, config.team_name_bar) for sample in frames]
    batch_method = getattr(reader, "read_text_batch", None)
    readings = batch_method(crops) if callable(batch_method) else [reader.read_text(crop) for crop in crops]
    observations = []
    for sample, (text, _) in zip(frames, readings):
        team, score = match_team_name(text, teams, config.team_similarity_threshold)
        observations.append(TeamObservation(sample.timestamp, text, team, score))
    return observations


def detect_team_boundary(
    observations: list[TeamObservation], debounce_frames: int = 2,
) -> list[TeamBoundary]:
    """Confirm a transition only after N consecutive frames agree on the new team."""
    if debounce_frames < 2:
        raise ValueError("debounce_frames must be >= 2")
    current: str | None = None
    current_last: float | None = None
    candidate: str | None = None
    candidate_start = 0.0
    streak = 0
    boundaries: list[TeamBoundary] = []
    for observation in sorted(observations, key=lambda item: item.timestamp):
        if observation.team is None:
            candidate, streak = None, 0
            continue
        if observation.team == candidate:
            streak += 1
        else:
            candidate, candidate_start, streak = observation.team, observation.timestamp, 1
        if observation.team == current:
            current_last = observation.timestamp
        if streak < debounce_frames or observation.team == current:
            continue
        if current is None:
            current, current_last = observation.team, observation.timestamp
            continue
        left = current_last if current_last is not None else candidate_start
        boundaries.append(TeamBoundary(
            current, observation.team, left, candidate_start, (left + candidate_start) / 2,
        ))
        current, current_last = observation.team, observation.timestamp
    return boundaries


def refine_team_boundary(
    video_path: str, boundary: TeamBoundary, teams: list[str], config: MappingConfig,
    reader: TextReader, *, frame_reader: Callable[[str, float], np.ndarray] = extract_frame,
) -> TeamBoundary:
    """Binary-search the last known old-team/first known new-team bracket."""
    low, high = boundary.left_seconds, boundary.right_seconds
    while high - low > config.boundary_precision_sec:
        middle = (low + high) / 2
        frame = frame_reader(video_path, middle)
        raw_text, _ = reader.read_text(crop_region(frame, config.team_name_bar))
        team, _ = match_team_name(raw_text, teams, config.team_similarity_threshold)
        if team == boundary.from_team:
            low = middle
        elif team == boundary.to_team:
            high = middle
        else:
            # An animation normally lies between the two stable labels. Probe both halves
            # on subsequent iterations while preserving a valid outer bracket.
            left_probe = (low + middle) / 2
            right_probe = (middle + high) / 2
            left_text, _ = reader.read_text(crop_region(
                frame_reader(video_path, left_probe), config.team_name_bar,
            ))
            right_text, _ = reader.read_text(crop_region(
                frame_reader(video_path, right_probe), config.team_name_bar,
            ))
            left_team, _ = match_team_name(left_text, teams, config.team_similarity_threshold)
            right_team, _ = match_team_name(right_text, teams, config.team_similarity_threshold)
            if left_team == boundary.from_team:
                low = left_probe
            if right_team == boundary.to_team:
                high = right_probe
            if left_team != boundary.from_team and right_team != boundary.to_team:
                break
    boundary.left_seconds = low
    boundary.right_seconds = high
    boundary.timestamp = (low + high) / 2
    return boundary
