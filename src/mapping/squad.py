from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mapping.schema import LineupRow, RawPlayer, SquadPlayer
from mapping.team import normalize_text, similarity


def load_squad(path: str | Path | None) -> list[SquadPlayer]:
    if path is None:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get(
            "squad", payload.get("squads", payload.get("squad_list", payload.get("players", []))),
        )
    if not isinstance(payload, list):
        raise ValueError("Squad JSON must be a list or contain squad/squads/players")
    result = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        result.append(SquadPlayer(
            str(item["team"]), int(item["jersey_number"]), str(item["player_name"]),
        ))
    return result


def _maximum_assignment(scores: list[list[float]]) -> list[int]:
    """Hungarian assignment for rows to unique columns, with appended dummy columns."""
    if not scores:
        return []
    row_count, real_columns = len(scores), len(scores[0]) if scores[0] else 0
    columns = real_columns + row_count
    maximum = max((value for row in scores for value in row), default=0.0)
    costs = [[maximum - value for value in row] + [maximum] * row_count for row in scores]
    u, v = [0.0] * (row_count + 1), [0.0] * (columns + 1)
    match, way = [0] * (columns + 1), [0] * (columns + 1)
    for row in range(1, row_count + 1):
        match[0] = row
        column = 0
        minimum = [float("inf")] * (columns + 1)
        used = [False] * (columns + 1)
        while True:
            used[column] = True
            current_row = match[column]
            delta, next_column = float("inf"), 0
            for candidate in range(1, columns + 1):
                if used[candidate]:
                    continue
                reduced = costs[current_row - 1][candidate - 1] - u[current_row] - v[candidate]
                if reduced < minimum[candidate]:
                    minimum[candidate], way[candidate] = reduced, column
                if minimum[candidate] < delta:
                    delta, next_column = minimum[candidate], candidate
            for candidate in range(columns + 1):
                if used[candidate]:
                    u[match[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if match[column] == 0:
                break
        while True:
            previous = way[column]
            match[column] = match[previous]
            column = previous
            if column == 0:
                break
    assigned = [-1] * row_count
    for column in range(1, columns + 1):
        if match[column] and column <= real_columns:
            assigned[match[column] - 1] = column - 1
    return assigned


def match_against_squad(
    observations: list[RawPlayer], team: str, squad: list[SquadPlayer],
    threshold: float = 72.0,
) -> list[tuple[int | None, str, str]]:
    candidates = [player for player in squad if normalize_text(player.team) == normalize_text(team)]
    if not candidates:
        return [(item.jersey_number, item.player_name, "low") for item in observations]
    scores: list[list[float]] = []
    for observation in observations:
        name_scores = [similarity(observation.player_name, player.player_name) for player in candidates]
        row = []
        for name_score, player in zip(name_scores, candidates):
            number_matches = observation.jersey_number == player.jersey_number
            if name_score >= threshold:
                row.append(name_score + (20.0 if number_matches else 0.0))
            elif number_matches:
                # A strong fuzzy name always outranks this number-only fallback, so
                # a confidently recognized name can correct a wrong OCR number.
                row.append(70.0)
            else:
                row.append(0.0)
        scores.append(row)
    assignment = _maximum_assignment(scores)
    result = []
    for row, observation in enumerate(observations):
        column = assignment[row]
        if column >= 0 and scores[row][column] > 0:
            player = candidates[column]
            result.append((player.jersey_number, player.player_name, "high"))
        else:
            result.append((observation.jersey_number, observation.player_name, "low"))
    return result
