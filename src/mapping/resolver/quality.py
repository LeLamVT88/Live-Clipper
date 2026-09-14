from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .common import normalize_text, similarity


UI_NAME_TOKENS = {
    "club", "coach", "formation", "league", "lineup", "substitute", "substitutes", "team",
}


@dataclass(frozen=True, slots=True)
class QualityResult:
    passed: bool
    resolved_players: int
    message: str


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
    expected_players: int, min_pair_confidence: float, expected_lineups: int = 1,
) -> QualityResult:
    if expected_players <= 0 or expected_lineups <= 0:
        raise ValueError("expected_players and expected_lineups must be positive.")
    if not 0 <= min_pair_confidence <= 1:
        raise ValueError("min_pair_confidence must be between 0 and 1.")
    resolved_players = len(records)
    reasons: list[str] = []
    status = str((diagnostic or {}).get("status", "")).strip().casefold()
    if status != "resolved":
        reasons.append("resolver status is not resolved")
    if not records:
        reasons.append("no resolved player rows")
    groups = _lineup_groups(records)
    if len(groups) != expected_lineups:
        reasons.append(f"found {len(groups)}/{expected_lineups} expected lineups")
    expected_total = expected_players * expected_lineups
    if resolved_players != expected_total:
        reasons.append(f"resolved {resolved_players}/{expected_total} expected players")
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
        weak_local_numbers: list[str] = []
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
            if str(record.get("number_source", "")) == "local_ocr":
                try:
                    evidence_frames = int(record.get("number_evidence_frames", 0))
                    name_evidence_frames = int(record.get("full_name_evidence_frames", 0))
                except (TypeError, ValueError):
                    evidence_frames = 0
                    name_evidence_frames = 0
                scout_confirmed_name = (
                    str(record.get("name_source", "")) == "scout"
                    and name_evidence_frames >= 2
                )
                if evidence_frames < 2 and not scout_confirmed_name:
                    weak_local_numbers.append(
                        player_name or f"slot {record.get('slot_index', '?')}"
                    )
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
        if weak_local_numbers:
            names = ", ".join(sorted(set(weak_local_numbers)))
            reasons.append(
                f"lineup {lineup_index} has local OCR shirt number(s) supported by "
                f"fewer than 2 frames: {names}"
            )
    if reasons:
        return QualityResult(False, resolved_players, "; ".join(dict.fromkeys(reasons)))
    return QualityResult(
        True, resolved_players,
        f"quality gate passed for {len(groups)} lineup(s), {resolved_players} player rows",
    )


def _candidate_groups(
    records: Iterable[dict[str, object]],
) -> list[list[dict[str, object]]]:
    groups: dict[tuple[str, int, int], list[dict[str, object]]] = {}
    for record in records:
        key = (
            str(record.get("video", "")),
            int(record.get("segment_index", 1)),
            int(record.get("lineup_index", 1)),
        )
        groups.setdefault(key, []).append(record)
    return list(groups.values())


def _same_team(
    left: list[dict[str, object]], right: list[dict[str, object]],
) -> bool:
    left_names = [str(row.get("player_name", "")) for row in left]
    right_names = [str(row.get("player_name", "")) for row in right]
    used: set[int] = set()
    matches = 0
    for left_name in left_names:
        match = next(
            (
                index for index, right_name in enumerate(right_names)
                if index not in used and similarity(left_name, right_name) >= 0.84
            ),
            None,
        )
        if match is not None:
            used.add(match)
            matches += 1
    return matches >= max(3, math.ceil(0.55 * min(len(left_names), len(right_names))))


def select_match_lineups(
    records: list[dict[str, object]], expected_lineups: int,
    expected_players: int,
) -> tuple[list[dict[str, object]], QualityResult]:
    """Select complete, distinct team lineups and assign match-wide indices."""
    candidates = [
        (order, rows)
        for order, rows in enumerate(_candidate_groups(records))
        if len(rows) == expected_players
    ]
    ranked = sorted(
        candidates,
        key=lambda item: -sum(float(row.get("pair_confidence", 0.0)) for row in item[1]),
    )
    selected: list[tuple[int, list[dict[str, object]]]] = []
    for candidate in ranked:
        if any(_same_team(candidate[1], chosen[1]) for chosen in selected):
            continue
        selected.append(candidate)
        if len(selected) == expected_lineups:
            break
    output: list[dict[str, object]] = []
    for lineup_index, (_, rows) in enumerate(sorted(selected), start=1):
        output.extend(dict(row, lineup_index=lineup_index) for row in rows)
    if len(selected) != expected_lineups:
        return output, QualityResult(
            False, len(output),
            f"match has {len(selected)}/{expected_lineups} distinct complete lineups",
        )
    return output, QualityResult(
        True, len(output),
        f"match quality gate passed for {expected_lineups} lineups, {len(output)} players",
    )
