from __future__ import annotations

import csv
import json
from pathlib import Path

from mapping.text import name_key
from mapping.validation import valid_complete_graphic


PLAYER_COLUMNS = [
    "video", "interval_index", "graphic_index", "layout", "slot_index",
    "shirt_number", "player_name", "name_confidence", "pair_confidence",
    "evidence_frames", "number_source",
]
DIAGNOSTIC_COLUMNS = [
    "video", "interval_index", "graphic_index", "start_seconds", "end_seconds",
    "best_frame_timestamp", "selected_frame_timestamps", "selection_tier", "layout",
    "status", "player_count", "confirmed_pair_count", "issues",
]


def export_results(extractions: list[dict[str, object]], output_directory: str | Path) -> dict[str, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    resolved_path = output / "resolved_starters.csv"
    partial_path = output / "partial_starters.csv"
    diagnostics_path = output / "lineup_diagnostics.csv"
    evidence_path = output / "lineup_evidence.json"

    player_rows = []
    partial_rows = []
    diagnostic_rows = []
    for interval_index, extraction in enumerate(extractions, 1):
        video = extraction["video_path"]
        seen_complete_lineups: set[tuple[tuple[str, int], ...]] = set()
        for graphic in extraction["graphics"]:
            confirmed = sum(bool(player["confirmed"]) for player in graphic["players"])
            names = [name_key(player["name"]) for player in graphic["players"]]
            numbers = [player["shirt_number"] for player in graphic["players"]]
            valid_complete = valid_complete_graphic(graphic)
            export_status = "complete" if valid_complete else "partial"
            diagnostic_issues = list(graphic["issues"])
            if graphic["status"] == "complete" and not valid_complete:
                diagnostic_issues.append("export_validation_failed")
            diagnostic_rows.append({
                "video": video,
                "interval_index": interval_index,
                "graphic_index": graphic["graphic_index"],
                "start_seconds": graphic["start_seconds"],
                "end_seconds": graphic["end_seconds"],
                "best_frame_timestamp": graphic["best_frame_timestamp"],
                "selected_frame_timestamps": json.dumps(graphic["selected_frame_timestamps"]),
                "selection_tier": graphic["selection_tier"],
                "layout": graphic["layout"],
                "status": export_status,
                "player_count": len(graphic["players"]),
                "confirmed_pair_count": confirmed,
                "issues": "|".join(dict.fromkeys(diagnostic_issues)),
            })
            destination = player_rows if valid_complete else partial_rows
            if valid_complete:
                signature = tuple(sorted(zip(names, numbers)))
                if signature in seen_complete_lineups:
                    continue
                seen_complete_lineups.add(signature)
                selected_players = graphic["players"]
            else:
                selected_players = [player for player in graphic["players"] if player["confirmed"]]
            for player in selected_players:
                destination.append({
                    "video": video,
                    "interval_index": interval_index,
                    "graphic_index": graphic["graphic_index"],
                    "layout": graphic["layout"],
                    "slot_index": player["slot_index"],
                    "shirt_number": player["shirt_number"],
                    "player_name": player["name"],
                    "name_confidence": player["name_confidence"],
                    "pair_confidence": player["pair_confidence"],
                    "evidence_frames": len(player["evidence_timestamps"]),
                    "number_source": player["number_source"],
                })

    for path, columns, rows in (
        (resolved_path, PLAYER_COLUMNS, player_rows),
        (partial_path, PLAYER_COLUMNS, partial_rows),
        (diagnostics_path, DIAGNOSTIC_COLUMNS, diagnostic_rows),
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    evidence_path.write_text(json.dumps({"extractions": extractions}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "resolved": resolved_path, "partial": partial_path,
        "diagnostics": diagnostics_path, "evidence": evidence_path,
    }
