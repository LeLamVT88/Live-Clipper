from __future__ import annotations

from collections import Counter, defaultdict
from statistics import median

from mapping.schema import FrameAnalysis, PlayerObservation
from mapping.text import name_key, names_compatible


def _center(observation: PlayerObservation) -> tuple[float, float]:
    box = observation.name_box
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _source_weight(observation: PlayerObservation) -> float:
    if observation.number_source == "inline_ocr":
        return 5.0
    if observation.panel_role == "starter_list" and observation.number_source == "spatial_ocr":
        return 4.0
    if observation.number_source == "spatial_ocr":
        return 3.0
    if observation.number_source == "local_preprocessed_ocr":
        return 1.0
    return .5


def _same_player_in_panels(group: list[PlayerObservation], observation: PlayerObservation) -> bool:
    if any(names_compatible(item.name, observation.name) for item in group):
        return True
    if observation.jersey_number is None:
        return False
    return any(item.jersey_number == observation.jersey_number
               and item.panel_role != observation.panel_role for item in group)


def _collapse_timestamp(observations: list[PlayerObservation]) -> list[PlayerObservation]:
    groups: list[list[PlayerObservation]] = []
    for observation in sorted(observations, key=lambda item: item.name_confidence, reverse=True):
        group = next((items for items in groups if _same_player_in_panels(items, observation)), None)
        if group is None:
            groups.append([observation])
        else:
            group.append(observation)
    collapsed = []
    for group in groups:
        best = max(group, key=lambda item: (
            item.panel_role == "starter_list", len(name_key(item.name)), item.name_confidence,
        ))
        values = {item.jersey_number for item in group if item.jersey_number is not None}
        merged = PlayerObservation(**{field: getattr(best, field) for field in (
            "timestamp", "panel_role", "name", "name_confidence", "name_box", "jersey_number",
            "number_confidence", "number_box", "number_source", "pair_confidence", "number_candidates",
        )})
        merged.number_candidates = sorted(set(value for item in group for value in item.number_candidates))
        if len(values) == 1:
            chosen = max((item for item in group if item.jersey_number is not None),
                         key=lambda item: (item.number_source == "inline_ocr", item.pair_confidence))
            merged.jersey_number = chosen.jersey_number
            merged.number_confidence = chosen.number_confidence
            merged.number_box = chosen.number_box
            merged.number_source = chosen.number_source
            merged.pair_confidence = chosen.pair_confidence
        elif len(values) > 1:
            ranked = sorted(
                ((sum(_source_weight(item) for item in group if item.jersey_number == value), value)
                 for value in values), reverse=True,
            )
            if ranked[0][0] >= ranked[1][0] * 1.5:
                chosen = max((item for item in group if item.jersey_number == ranked[0][1]),
                             key=_source_weight)
                merged.jersey_number = chosen.jersey_number
                merged.number_confidence = chosen.number_confidence
                merged.number_box = chosen.number_box
                merged.number_source = chosen.number_source
                merged.pair_confidence = chosen.pair_confidence
            else:
                merged.jersey_number = None
                merged.number_source = "same_frame_conflict"
                merged.pair_confidence = 0.0
        collapsed.append(merged)
    return collapsed


def resolve_frames(frames: list[FrameAnalysis]) -> tuple[list[dict[str, object]], list[str]]:
    tracks: list[list[PlayerObservation]] = []
    for frame in sorted(frames, key=lambda item: item.timestamp):
        for observation in _collapse_timestamp(frame.starter_observations):
            x, y = _center(observation)
            choices = []
            for index, track in enumerate(tracks):
                if any(item.timestamp == observation.timestamp for item in track):
                    continue
                tx = median(_center(item)[0] for item in track)
                ty = median(_center(item)[1] for item in track)
                compatible = any(names_compatible(item.name, observation.name) for item in track)
                if compatible:
                    choices.append((abs(tx - x) + abs(ty - y), index))
            if choices:
                tracks[min(choices)[1]].append(observation)
            else:
                tracks.append([observation])

    players = []
    for track in tracks:
        timestamps = sorted({item.timestamp for item in track})
        if len(timestamps) < 2:
            continue
        best_name = max(track, key=lambda item: item.name_confidence)
        votes: dict[int, set[float]] = defaultdict(set)
        vote_scores: Counter[int] = Counter()
        nonlocal_scores: Counter[int] = Counter()
        candidate_times: dict[int, set[float]] = defaultdict(set)
        sole_candidate_times: dict[int, set[float]] = defaultdict(set)
        sources: dict[int, Counter[str]] = defaultdict(Counter)
        candidate_values = set()
        for item in track:
            candidate_values.update(item.number_candidates)
            for candidate in set(item.number_candidates):
                candidate_times[candidate].add(item.timestamp)
            if len(set(item.number_candidates)) == 1:
                sole_candidate_times[item.number_candidates[0]].add(item.timestamp)
            if item.jersey_number is not None:
                votes[item.jersey_number].add(item.timestamp)
                vote_scores[item.jersey_number] += _source_weight(item)
                if item.number_source != "local_preprocessed_ocr":
                    nonlocal_scores[item.jersey_number] += _source_weight(item)
                sources[item.jersey_number][item.number_source] += 1
        strong = sorted(
            ((vote_scores[number], len(seen), number) for number, seen in votes.items() if len(seen) >= 2),
            reverse=True,
        )
        number = None
        number_source = None
        if strong and (len(strong) == 1 or (
                nonlocal_scores[strong[0][2]] > 0 and strong[0][0] >= strong[1][0] * 1.6)):
            number = strong[0][2]
            number_source = sources[number].most_common(1)[0][0]
        if number is None:
            weak = sorted(
                ((len(seen), len(sole_candidate_times[candidate]), candidate)
                 for candidate, seen in candidate_times.items()), reverse=True,
            )
            if weak and len(weak) == 1 and weak[0][0] >= 4 and weak[0][1] >= 2:
                number = weak[0][2]
                number_source = "local_candidate_consensus"
        conflict = number is None and (
            len(votes) > 1 or any(item.number_source == "same_frame_conflict" for item in track)
        )
        name_confirmed = len(timestamps) >= 2
        number_confirmed = number is not None
        x = median(_center(item)[0] for item in track)
        y = median(_center(item)[1] for item in track)
        players.append({
            "name": best_name.name,
            "shirt_number": number,
            "name_confirmed": name_confirmed,
            "number_confirmed": number_confirmed,
            "confirmed": name_confirmed and number_confirmed,
            "name_confidence": round(sum(item.name_confidence for item in track) / len(track), 4),
            "pair_confidence": round(sum(item.pair_confidence for item in track) / len(track), 4),
            "number_status": "conflict" if conflict else (
                "weighted_consensus" if number_confirmed else "insufficient_evidence"
            ),
            "number_candidates": sorted(candidate_values | set(votes)),
            "number_source": number_source,
            "evidence_timestamps": [round(value, 3) for value in timestamps],
            "normalized_center": [round(x, 5), round(y, 5)],
            "observations": [item.to_evidence() for item in track],
        })

    players.sort(key=lambda item: (item["normalized_center"][1], item["normalized_center"][0]))
    for slot, player in enumerate(players, 1):
        player["slot_index"] = slot

    issues = []
    if len(players) != 11:
        issues.append(f"resolved_player_count_{len(players)}")
    duplicate_names = [name for name, count in Counter(name_key(p["name"]) for p in players).items() if count > 1]
    if duplicate_names:
        issues.append("duplicate_names")
    numbers = [p["shirt_number"] for p in players if p["shirt_number"] is not None]
    duplicates = {number for number, count in Counter(numbers).items() if count > 1}
    if duplicates:
        for player in players:
            if player["shirt_number"] in duplicates:
                player["number_candidates"] = sorted(set(player["number_candidates"]) | {player["shirt_number"]})
                player["shirt_number"] = None
                player["number_confirmed"] = player["confirmed"] = False
                player["number_status"] = "duplicate_conflict"
        issues.append("duplicate_shirt_numbers")
    if any(not p["name_confirmed"] for p in players):
        issues.append("unconfirmed_names")
    if any(not p["number_confirmed"] for p in players):
        issues.append("unconfirmed_numbers")
    return players, list(dict.fromkeys(issues))
