from __future__ import annotations

from mapping.text import name_key, names_compatible
from mapping.validation import refresh_graphic


def _name_overlap(left: dict[str, object], right: dict[str, object]) -> int:
    unmatched = list(right["players"])
    matches = 0
    for player in left["players"]:
        match = next((candidate for candidate in unmatched
                      if names_compatible(player["name"], candidate["name"])), None)
        if match is not None:
            unmatched.remove(match)
            matches += 1
    return matches


def _quality(graphic: dict[str, object]) -> tuple[object, ...]:
    players = graphic["players"]
    return (
        graphic["status"] == "complete",
        -abs(len(players) - 11),
        sum(bool(player["confirmed"]) for player in players),
        "list" in graphic["layout"],
        len(players),
    )


def _copy_player(player: dict[str, object]) -> dict[str, object]:
    return player.copy() | {
        "number_candidates": list(player["number_candidates"]),
        "evidence_timestamps": list(player["evidence_timestamps"]),
        "observations": list(player["observations"]),
    }


def _merge_player(target: dict[str, object], source: dict[str, object]) -> None:
    if names_compatible(target["name"], source["name"]):
        if len(name_key(source["name"])) > len(name_key(target["name"])):
            target["name"] = source["name"]
    if target["shirt_number"] is None and source["number_confirmed"]:
        for key in ("shirt_number", "number_confirmed", "number_status",
                    "number_source", "pair_confidence"):
            target[key] = source[key]
    target["confirmed"] = bool(target["name_confirmed"] and target["number_confirmed"])
    target["number_candidates"] = sorted(set(target["number_candidates"]) | set(source["number_candidates"]))
    target["evidence_timestamps"] = sorted(set(target["evidence_timestamps"])
                                               | set(source["evidence_timestamps"]))
    target["observations"].extend(source["observations"])


def _merge_cluster(cluster: list[dict[str, object]]) -> dict[str, object]:
    primary = max(cluster, key=_quality)
    merged = primary.copy()
    merged["players"] = [_copy_player(player) for player in primary["players"]]
    for phase in cluster:
        if phase is primary:
            continue
        for source in phase["players"]:
            target = next((player for player in merged["players"]
                           if names_compatible(player["name"], source["name"])), None)
            if target is not None:
                _merge_player(target, source)
            elif len(merged["players"]) < 11:
                merged["players"].append(_copy_player(source))

    merged["start_seconds"] = min(phase["start_seconds"] for phase in cluster)
    merged["end_seconds"] = max(phase["end_seconds"] for phase in cluster)
    merged["selected_frame_timestamps"] = sorted({timestamp for phase in cluster
                                                  for timestamp in phase["selected_frame_timestamps"]})
    merged["selection_tier"] = max(phase["selection_tier"] for phase in cluster)
    roles = sorted({role for phase in cluster for role in phase["panel_roles"]})
    merged["panel_roles"] = roles
    merged["layout"] = "+".join(role.removeprefix("starter_") for role in roles) or "unknown"
    merged["local_number_evidence"] = [item for phase in cluster
                                       for item in phase["local_number_evidence"]]
    merged["ocr_evidence"] = [item for phase in cluster for item in phase["ocr_evidence"]]
    refresh_graphic(merged)
    return merged


def consolidate_graphics(graphics: list[dict[str, object]]) -> list[dict[str, object]]:
    """Merge animation phases only when most of the shorter lineup agrees."""
    clusters: list[list[dict[str, object]]] = []
    for graphic in graphics:
        cluster = None
        for items in reversed(clusters):
            shorter = min(len(items[-1]["players"]), len(graphic["players"]))
            overlap = _name_overlap(items[-1], graphic)
            if shorter >= 6 and overlap >= min(7, shorter) and overlap / shorter >= .70:
                cluster = items
                break
        (clusters.append([graphic]) if cluster is None else cluster.append(graphic))
    result = [_merge_cluster(cluster) for cluster in clusters]
    for index, graphic in enumerate(result, 1):
        graphic["graphic_index"] = index
    return result
