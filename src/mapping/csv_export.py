from __future__ import annotations

import csv
import json
from pathlib import Path


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
    diagnostics_path = output / "lineup_diagnostics.csv"
    evidence_path = output / "lineup_evidence.json"

    player_rows = []
    diagnostic_rows = []
    for interval_index, extraction in enumerate(extractions, 1):
        video = extraction["video_path"]
        for graphic in extraction["graphics"]:
            confirmed = sum(bool(player["confirmed"]) for player in graphic["players"])
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
                "status": graphic["status"],
                "player_count": len(graphic["players"]),
                "confirmed_pair_count": confirmed,
                "issues": "|".join(graphic["issues"]),
            })
            if graphic["status"] != "complete":
                continue
            for player in graphic["players"]:
                player_rows.append({
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
        (diagnostics_path, DIAGNOSTIC_COLUMNS, diagnostic_rows),
    ):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    evidence_path.write_text(json.dumps({"extractions": extractions}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"resolved": resolved_path, "diagnostics": diagnostics_path, "evidence": evidence_path}
