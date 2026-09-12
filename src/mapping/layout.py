from __future__ import annotations

import numpy as np

from mapping.assignment import pair_panel
from mapping.cards import build_player_cards, detect_player_panels
from mapping.exclusions import detect_exclusions, outside_exclusions
from mapping.schema import BBox, FrameAnalysis, OCRToken
from mapping.text import is_likely_metadata, is_player_name, names_compatible


def _bbox(points: object, width: int, height: int) -> BBox:
    array = np.asarray(points, dtype=float).reshape(-1, 2)
    if array.size == 0 or width <= 0 or height <= 0:
        raise ValueError("OCR polygon and frame dimensions must be non-empty")
    return (
        float(np.clip(array[:, 0].min() / width, 0, 1)),
        float(np.clip(array[:, 1].min() / height, 0, 1)),
        float(np.clip(array[:, 0].max() / width, 0, 1)),
        float(np.clip(array[:, 1].max() / height, 0, 1)),
    )


def make_tokens(
    texts: list[str], scores: list[float], polygons: list, frame_shape: tuple[int, ...],
) -> list[OCRToken]:
    if not len(texts) == len(scores) == len(polygons):
        raise ValueError("OCR text/confidence/polygon counts do not match")
    height, width = frame_shape[:2]
    return [OCRToken(index, str(text).strip(), float(score), _bbox(box, width, height))
            for index, (text, score, box) in enumerate(zip(texts, scores, polygons))
            if str(text).strip()]


def _unique_observations(observations: list) -> list:
    unique = []
    for observation in sorted(observations, key=lambda item: item.name_confidence, reverse=True):
        if not any(names_compatible(observation.name, existing.name) for existing in unique):
            unique.append(observation)
    return unique


def analyze_layout(
    texts: list[str], scores: list[float], polygons: list, frame_shape: tuple[int, ...],
    timestamp: float, sharpness: float,
) -> FrameAnalysis:
    """Convert full-frame OCR into exclusions, visual cards and starter observations."""
    tokens = make_tokens(texts, scores, polygons, frame_shape)
    exclusions, semantic_panels = detect_exclusions(tokens)
    assignable_tokens = outside_exclusions(tokens, exclusions)
    names = [token for token in assignable_tokens if token.confidence >= .45
             and not is_likely_metadata(token.text) and is_player_name(token.text)]
    player_panels = detect_player_panels(names, assignable_tokens)
    panels = [*semantic_panels, *player_panels]

    observations = [observation for panel in player_panels if panel.role.startswith("starter_")
                    for observation in pair_panel(panel, assignable_tokens, timestamp)]
    unique = _unique_observations(observations)
    cards = build_player_cards(player_panels, unique)
    weak_count = sum(len(panel.name_tokens) for panel in player_panels if panel.role == "weak_pitch")
    candidate_count = len(unique) or weak_count
    paired = sum(observation.jersey_number is not None for observation in unique)

    issues = []
    if not unique:
        issues.append("no_starter_panel")
    if unique and not 7 <= candidate_count <= 13:
        issues.append(f"candidate_player_count_{candidate_count}")
    score = (max(0.0, 1.0 - abs(candidate_count - 11) / 11)
             + min(1.0, paired / 11) + min(.2, sharpness / 5000))
    return FrameAnalysis(
        timestamp, panels, observations, sharpness, score, issues,
        {
            "texts": texts,
            "confidences": [round(float(value), 5) for value in scores],
            "polygons": polygons,
            "frame_shape": list(frame_shape),
        },
        cards=cards,
        exclusions=exclusions,
    )
