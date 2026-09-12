from __future__ import annotations

from mapping.text import is_likely_metadata, name_key


def player_issues(players: list[dict[str, object]]) -> list[str]:
    issues: list[str] = []
    names = [name_key(player["name"]) for player in players]
    confirmed_numbers = [player["shirt_number"] for player in players
                         if player["number_confirmed"]]
    if len(players) != 11:
        issues.append(f"resolved_player_count_{len(players)}")
    if len(set(names)) != len(names):
        issues.append("duplicate_names")
    if len(set(confirmed_numbers)) != len(confirmed_numbers):
        issues.append("duplicate_shirt_numbers")
    if any(is_likely_metadata(player["name"]) for player in players):
        issues.append("metadata_in_starters")
    if any(not player["name_confirmed"] for player in players):
        issues.append("unconfirmed_names")
    if any(not player["number_confirmed"] for player in players):
        issues.append("unconfirmed_numbers")
    return issues


def refresh_graphic(graphic: dict[str, object]) -> None:
    for slot, player in enumerate(graphic["players"], 1):
        player["slot_index"] = slot
    graphic["issues"] = player_issues(graphic["players"])
    graphic["status"] = "complete" if not graphic["issues"] else "partial"


def extraction_status(graphics: list[dict[str, object]]) -> str:
    if not graphics:
        return "no_stable_graphic"
    complete = sum(graphic["status"] == "complete" for graphic in graphics)
    if complete == len(graphics):
        return "complete"
    return "mixed" if complete else "partial"


def valid_complete_graphic(graphic: dict[str, object]) -> bool:
    players = graphic["players"]
    names = [name_key(player["name"]) for player in players]
    numbers = [player["shirt_number"] for player in players]
    return (
        graphic["status"] == "complete" and len(players) == 11
        and len(set(names)) == 11 and all(names)
        and all(number is not None for number in numbers) and len(set(numbers)) == 11
        and all(player["confirmed"] for player in players)
        and not any(is_likely_metadata(player["name"]) for player in players)
    )
