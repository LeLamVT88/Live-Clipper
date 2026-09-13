from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .common import normalize_text


UI_NAME_TOKENS = {
    "club", "coach", "formation", "league", "lineup", "substitute", "substitutes", "team",
}


@dataclass(frozen=True, slots=True)
class QualityResult:
    passed: bool
    resolved_players: int
    message: str


def _diagnostic_status(diagnostic: dict[str, object] | None) -> str:
    if diagnostic is None:
        return ""
    return str(diagnostic.get("status", "")).strip().casefold()


def _lineup_groups(records: Iterable[dict[str, object]]) -> dict[int, list[dict[str, object]]]:
    groups: dict[int, list[dict[str, object]]] = {}
    for record in records:
        try:
            lineup_index = int(record.get("lineup_index", 1))
        except (TypeError, ValueError):
            lineup_index = 1
        groups.setdefault(lineup_index, []).append(record)
    return groups


def evaluate_segment_quality(
    records: list[dict[str, object]], diagnostic: dict[str, object] | None,
    expected_players: int, min_pair_confidence: float,
) -> QualityResult:
    if expected_players <= 0:
        raise ValueError("expected_players must be positive.")
    if not 0 <= min_pair_confidence <= 1:
        raise ValueError("min_pair_confidence must be between 0 and 1.")
    resolved_players = len(records)
    reasons: list[str] = []
    if _diagnostic_status(diagnostic) != "resolved":
        reasons.append("resolver status is not resolved")
    if not records:
        reasons.append("no resolved player rows")
    groups = _lineup_groups(records)
    for lineup_index, lineup_records in sorted(groups.items()):
        if len(lineup_records) != expected_players:
            reasons.append(f"lineup {lineup_index} has {len(lineup_records)}/{expected_players} players")
        shirt_numbers: list[int] = []
        for record in lineup_records:
            try:
                shirt_numbers.append(int(record.get("shirt_number")))
            except (TypeError, ValueError):
                continue
        if len(shirt_numbers) != len(lineup_records):
            reasons.append(f"lineup {lineup_index} has missing shirt numbers")
        elif len(set(shirt_numbers)) != expected_players:
            reasons.append(
                f"lineup {lineup_index} has {len(set(shirt_numbers))}/{expected_players} unique shirt numbers"
            )
        empty_names = 0
        ui_names: list[str] = []
        low_confidence: list[str] = []
        for record in lineup_records:
            player_name = str(record.get("player_name", "")).strip()
            normalized_name = normalize_text(player_name)
            if not normalized_name:
                empty_names += 1
            elif UI_NAME_TOKENS.intersection(normalized_name.split()):
                ui_names.append(player_name)
            try:
                pair_confidence = float(record.get("pair_confidence"))
            except (TypeError, ValueError):
                pair_confidence = -1.0
            if not math.isfinite(pair_confidence) or pair_confidence < min_pair_confidence:
                low_confidence.append(player_name or f"slot {record.get('slot_index', '?')}")
        if empty_names:
            reasons.append(f"lineup {lineup_index} has {empty_names} empty player name(s)")
        if ui_names:
            names = ", ".join(sorted(set(ui_names)))
            reasons.append(f"lineup {lineup_index} contains UI text as player name: {names}")
        if low_confidence:
            reasons.append(
                f"lineup {lineup_index} has {len(low_confidence)} pair(s) below confidence "
                f"{min_pair_confidence:.2f}"
            )
    if reasons:
        return QualityResult(False, resolved_players, "; ".join(dict.fromkeys(reasons)))
    return QualityResult(
        True, resolved_players,
        f"quality gate passed for {len(groups)} lineup(s), {resolved_players} player rows",
    )
