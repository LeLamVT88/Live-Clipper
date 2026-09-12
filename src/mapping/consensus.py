from __future__ import annotations

from mapping.card_consensus import (
    recover_goalkeeper_one, resolve_duplicate_numbers, resolve_track,
)
from mapping.schema import FrameAnalysis
from mapping.tracking import build_card_tracks
from mapping.validation import player_issues


def resolve_frames(frames: list[FrameAnalysis]) -> tuple[list[dict[str, object]], list[str]]:
    """Resolve independent player-card tracks and then enforce lineup-wide rules."""
    players = [player for track in build_card_tracks(frames)
               if (player := resolve_track(track)) is not None]
    players.sort(key=lambda item: (item["normalized_center"][1], item["normalized_center"][0]))
    for slot, player in enumerate(players, 1):
        player["slot_index"] = slot

    recover_goalkeeper_one(players)
    had_duplicates = resolve_duplicate_numbers(players)
    issues = player_issues(players)
    if had_duplicates:
        issues.insert(1 if issues else 0, "duplicate_shirt_numbers")
    return players, list(dict.fromkeys(issues))
