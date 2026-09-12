from __future__ import annotations

from statistics import median

from mapping.schema import FrameAnalysis, PlayerObservation
from mapping.text import name_key, names_compatible


def center(observation: PlayerObservation) -> tuple[float, float]:
    x0, y0, x1, y1 = observation.name_box
    return (x0 + x1) / 2, (y0 + y1) / 2


def source_weight(observation: PlayerObservation) -> float:
    if observation.number_source == "inline_ocr":
        return 5.0
    if observation.panel_role == "starter_list" and observation.number_source == "spatial_ocr":
        return 4.0
    if observation.number_source == "spatial_ocr":
        return 3.0
    return 1.0 if observation.number_source == "local_preprocessed_ocr" else .5


def supplement_scout_names(detailed: FrameAnalysis, scout: FrameAnalysis | None) -> None:
    """Keep visual card slots that tiny OCR saw but detailed OCR happened to miss."""
    if scout is None:
        return
    existing = detailed.starter_observations
    for source in scout.starter_observations:
        if any(names_compatible(source.name, item.name) for item in existing):
            continue
        added = PlayerObservation(
            timestamp=detailed.timestamp,
            panel_role=source.panel_role,
            name=source.name,
            name_confidence=source.name_confidence,
            name_box=source.name_box,
        )
        detailed.observations.append(added)
        existing.append(added)


def _same_card(group: list[PlayerObservation], observation: PlayerObservation) -> bool:
    if any(names_compatible(item.name, observation.name) for item in group):
        return True
    return observation.jersey_number is not None and any(
        item.jersey_number == observation.jersey_number
        and item.panel_role != observation.panel_role for item in group
    )


def _copy_observation(observation: PlayerObservation) -> PlayerObservation:
    return PlayerObservation(**{
        field: getattr(observation, field) for field in (
            "timestamp", "panel_role", "name", "name_confidence", "name_box",
            "jersey_number", "number_confidence", "number_box", "number_source",
            "pair_confidence", "number_candidates", "number_candidate_scores",
        )
    })


def collapse_timestamp(observations: list[PlayerObservation]) -> list[PlayerObservation]:
    groups: list[list[PlayerObservation]] = []
    for observation in sorted(observations, key=lambda item: item.name_confidence, reverse=True):
        group = next((items for items in groups if _same_card(items, observation)), None)
        groups.append([observation]) if group is None else group.append(observation)

    collapsed: list[PlayerObservation] = []
    for group in groups:
        best = max(group, key=lambda item: (
            item.panel_role == "starter_list", len(name_key(item.name)), item.name_confidence,
        ))
        merged = _copy_observation(best)
        merged.number_candidates = sorted({candidate for item in group
                                           for candidate in item.number_candidates})
        merged.number_candidate_scores = {
            candidate: max(item.number_candidate_scores.get(candidate, 0.0) for item in group)
            for candidate in merged.number_candidates
        }
        numbered = [item for item in group if item.jersey_number is not None]
        values = {item.jersey_number for item in numbered}
        if len(values) == 1:
            chosen = max(numbered, key=lambda item: (source_weight(item), item.pair_confidence))
            for field in ("jersey_number", "number_confidence", "number_box",
                          "number_source", "pair_confidence"):
                setattr(merged, field, getattr(chosen, field))
        elif len(values) > 1:
            ranked = sorted((sum(source_weight(item) for item in numbered
                                 if item.jersey_number == value), value)
                            for value in values)
            if ranked[-1][0] >= ranked[-2][0] * 1.5:
                chosen = max((item for item in numbered if item.jersey_number == ranked[-1][1]),
                             key=source_weight)
                for field in ("jersey_number", "number_confidence", "number_box",
                              "number_source", "pair_confidence"):
                    setattr(merged, field, getattr(chosen, field))
            else:
                merged.jersey_number, merged.number_source, merged.pair_confidence = None, "same_frame_conflict", 0.0
        collapsed.append(merged)
    return collapsed


def build_card_tracks(frames: list[FrameAnalysis]) -> list[list[PlayerObservation]]:
    tracks: list[list[PlayerObservation]] = []
    for frame in sorted(frames, key=lambda item: item.timestamp):
        for observation in collapse_timestamp(frame.starter_observations):
            choices = []
            x, y = center(observation)
            for index, track in enumerate(tracks):
                if any(item.timestamp == observation.timestamp for item in track):
                    continue
                if any(names_compatible(item.name, observation.name) for item in track):
                    tx = median(center(item)[0] for item in track)
                    ty = median(center(item)[1] for item in track)
                    choices.append((abs(tx - x) + abs(ty - y), index))
            tracks[min(choices)[1]].append(observation) if choices else tracks.append([observation])
    return tracks
