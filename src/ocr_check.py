from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import ensure_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "crop_metadata.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "ocr_check.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare an OCR check CSV from crop metadata. OCR is not implemented yet."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_csv.exists():
        print(f"Input CSV does not exist: {args.input_csv}")
        print("Run python src/localize_crop.py with crop coordinates first.")
        return 1

    crops = pd.read_csv(args.input_csv)
    required_columns = {"video", "crop_path", "timestamp", "timestamp_seconds"}
    missing = sorted(required_columns - set(crops.columns))
    if missing:
        print(f"Missing columns in {args.input_csv}: {', '.join(missing)}")
        return 1

    checks = crops[["video", "crop_path", "timestamp", "timestamp_seconds"]].copy()
    checks["ocr_text"] = ""
    checks["needs_review"] = True

    ensure_dir(args.output_csv.parent)
    checks.to_csv(args.output_csv, index=False)
    print(f"Saved OCR review CSV to: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
