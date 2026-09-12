from __future__ import annotations

import argparse
import logging
from pathlib import Path

from mapping.config import MappingConfig, load_config
from mapping.pipeline import extract_lineup
from mapping.squad import load_squad


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract football starting-lineup shirt numbers and names to CSV",
    )
    parser.add_argument("--video", type=Path, required=True, help="Short MP4 containing one or two team lineups")
    parser.add_argument("--teams", required=True, help='Official team names separated by comma, e.g. "Belgium,France"')
    parser.add_argument("--squad-json", type=Path, help="Optional registered squad metadata")
    parser.add_argument("--config", type=Path, help="JSON/YAML layout config (defaults to bundled config)")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--ocr-backend", choices=("paddleocr", "llm_vision"))
    parser.add_argument("--no-split-clips", action="store_true", help="Do not write physical clips at a two-team boundary")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    teams = [team.strip() for team in args.teams.split(",") if team.strip()]
    config = load_config(args.config)
    if args.ocr_backend or args.no_split_clips:
        values = {field: getattr(config, field) for field in config.__dataclass_fields__}
        if args.ocr_backend:
            ocr_values = {field: getattr(config.ocr, field) for field in config.ocr.__dataclass_fields__}
            ocr_values["backend"] = args.ocr_backend
            values["ocr"] = type(config.ocr)(**ocr_values)
        if args.no_split_clips:
            values["write_split_clips"] = False
        config = MappingConfig(**values)
    rows = extract_lineup(
        args.video, teams, squad=load_squad(args.squad_json), config=config,
        output_csv=args.out, split_directory=args.out.parent / f"{args.video.stem}_segments",
    )
    print(f"Wrote {len(rows)} starting-player row(s) to {args.out}")


if __name__ == "__main__":
    main()

