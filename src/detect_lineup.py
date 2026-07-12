from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import ensure_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRAME_LABELS_CSV = PROJECT_ROOT / "data" / "processed" / "frame_labels.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "outputs" / "predictions" / "lineup_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a simple baseline prediction CSV for lineup detection."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_FRAME_LABELS_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_csv.exists():
        print(f"Input CSV does not exist: {args.input_csv}")
        print("Run python src/build_frame_labels.py first.")
        return 1

    frames = pd.read_csv(args.input_csv)
    required_columns = {"video", "frame_path", "timestamp", "timestamp_seconds"}
    missing = sorted(required_columns - set(frames.columns))
    if missing:
        print(f"Missing columns in {args.input_csv}: {', '.join(missing)}")
        return 1

    # Temporary baseline: if label exists, copy it as score; otherwise score all frames as 0.
    predictions = frames[["video", "frame_path", "timestamp", "timestamp_seconds"]].copy()
    predictions["score"] = frames["label"].astype(float) if "label" in frames else 0.0
    predictions["pred_label"] = (predictions["score"] >= 0.5).astype(int)

    ensure_dir(args.output_csv.parent)
    predictions.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
