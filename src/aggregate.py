from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import ensure_dir, seconds_to_timestamp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS_CSV = PROJECT_ROOT / "outputs" / "predictions" / "lineup_predictions.csv"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "outputs" / "predictions" / "lineup_segments.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate frame-level lineup predictions into time segments."
    )
    parser.add_argument("--predictions-csv", type=Path, default=DEFAULT_PREDICTIONS_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.predictions_csv.exists():
        print(f"Predictions CSV does not exist: {args.predictions_csv}")
        print("Run python src/detect_lineup.py first.")
        return 1

    predictions = pd.read_csv(args.predictions_csv)
    required_columns = {"video", "timestamp_seconds"}
    missing = sorted(required_columns - set(predictions.columns))
    if missing:
        print(f"Missing columns in {args.predictions_csv}: {', '.join(missing)}")
        return 1

    score_column = "score" if "score" in predictions.columns else "pred_label"
    if score_column not in predictions.columns:
        print("Predictions CSV must contain either score or pred_label.")
        return 1

    segments: list[dict[str, object]] = []
    for video, group in predictions.sort_values(["video", "timestamp_seconds"]).groupby("video"):
        active_start: float | None = None
        active_end: float | None = None

        for _, row in group.iterrows():
            timestamp_seconds = float(row["timestamp_seconds"])
            is_lineup = float(row[score_column]) >= args.threshold

            if is_lineup and active_start is None:
                active_start = timestamp_seconds
            if is_lineup:
                active_end = timestamp_seconds
            if not is_lineup and active_start is not None and active_end is not None:
                segments.append(
                    {
                        "video": video,
                        "start": seconds_to_timestamp(active_start),
                        "end": seconds_to_timestamp(active_end),
                        "start_seconds": active_start,
                        "end_seconds": active_end,
                    }
                )
                active_start = None
                active_end = None

        if active_start is not None and active_end is not None:
            segments.append(
                {
                    "video": video,
                    "start": seconds_to_timestamp(active_start),
                    "end": seconds_to_timestamp(active_end),
                    "start_seconds": active_start,
                    "end_seconds": active_end,
                }
            )

    ensure_dir(args.output_csv.parent)
    pd.DataFrame(
        segments,
        columns=["video", "start", "end", "start_seconds", "end_seconds"],
    ).to_csv(args.output_csv, index=False)
    print(f"Saved segments to: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
