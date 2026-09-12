from __future__ import annotations

import csv
from pathlib import Path

from mapping.schema import LineupRow


CSV_COLUMNS = [
    "team", "jersey_number", "player_name", "role", "source_clip",
    "frame_timestamp_sec", "ocr_confidence",
]


def write_csv(rows: list[LineupRow], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "team": row.team,
                "jersey_number": "" if row.jersey_number is None else row.jersey_number,
                "player_name": row.player_name,
                "role": row.role,
                "source_clip": row.source_clip,
                "frame_timestamp_sec": f"{row.frame_timestamp_sec:.3f}",
                "ocr_confidence": row.ocr_confidence,
            })
    return path
