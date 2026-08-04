from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import ensure_dir, seconds_to_timestamp


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions" / "mobilenet"
DEFAULT_PREDICTIONS_CSV = (
    DEFAULT_OUTPUT_DIR / "mobilenet_v3_small_inference.csv"
)
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "lineup_segments.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate frame-level lineup predictions into time segments."
    )
    parser.add_argument("--predictions-csv", type=Path, default=DEFAULT_PREDICTIONS_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Override the decision threshold. By default, use pred_label produced "
            "with the threshold saved in the MobileNet checkpoint."
        ),
    )
    parser.add_argument(
        "--merge-gap-seconds",
        type=float,
        default=0.0,
        help="Merge predicted segments separated by this many seconds or less.",
    )
    parser.add_argument(
        "--min-duration-seconds",
        type=float,
        default=0.0,
        help="Drop predicted segments shorter than this duration.",
    )
    parser.add_argument(
        "--expected-segments-per-video",
        type=int,
        default=None,
        help=(
            "Force this many segments per video by splitting at local score "
            "valleys or merging the closest segments."
        ),
    )
    return parser.parse_args()


def append_segment(
    segments: list[dict[str, object]],
    video: str,
    start_seconds: float,
    end_seconds: float,
) -> None:
    segments.append(
        {
            "video": video,
            "start": seconds_to_timestamp(start_seconds),
            "end": seconds_to_timestamp(end_seconds),
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
        }
    )


def infer_frame_duration(timestamps: np.ndarray) -> float:
    """Infer how long the final sampled frame represents."""
    if len(timestamps) < 2:
        return 0.0

    differences = np.diff(timestamps)
    positive_differences = differences[differences > 0]
    if len(positive_differences) == 0:
        return 0.0
    return float(np.median(positive_differences))


def predictions_to_segments(
    predictions: pd.DataFrame,
    threshold: float | None = None,
) -> list[dict[str, object]]:
    """Convert sampled predictions to half-open lineup intervals [start, end)."""
    required_columns = {"video", "timestamp_seconds"}
    missing = sorted(required_columns - set(predictions.columns))
    if missing:
        raise ValueError(
            f"Predictions are missing required columns: {', '.join(missing)}"
        )

    if threshold is None:
        if "pred_label" not in predictions.columns:
            raise ValueError(
                "Predictions must contain pred_label when threshold is omitted."
            )
        score_column = "pred_label"
        decision_threshold = 0.5
    else:
        score_column = next(
            (
                column
                for column in ("smoothed_score", "score", "pred_label")
                if column in predictions.columns
            ),
            None,
        )
        if score_column is None:
            raise ValueError(
                "Predictions must contain smoothed_score, score, or pred_label."
            )
        decision_threshold = threshold

    segments: list[dict[str, object]] = []
    ordered = predictions.sort_values(["video", "timestamp_seconds"])
    for video, group in ordered.groupby("video", sort=False):
        timestamps = pd.to_numeric(
            group["timestamp_seconds"], errors="raise"
        ).to_numpy(dtype=float)
        if len(timestamps) == 0:
            continue
        if not np.isfinite(timestamps).all() or (timestamps < 0).any():
            raise ValueError(f"Video {video} has invalid timestamp_seconds values.")
        if len(timestamps) > 1 and (np.diff(timestamps) <= 0).any():
            raise ValueError(
                f"Video {video} must have unique, strictly increasing timestamps."
            )

        scores = pd.to_numeric(group[score_column], errors="raise").to_numpy(
            dtype=float
        )
        is_lineup = scores >= decision_threshold
        final_frame_duration = infer_frame_duration(timestamps)
        active_start: float | None = None

        for position, (timestamp_seconds, lineup_at_timestamp) in enumerate(
            zip(timestamps, is_lineup, strict=True)
        ):
            if lineup_at_timestamp and active_start is None:
                active_start = float(timestamp_seconds)
            if not lineup_at_timestamp and active_start is not None:
                append_segment(
                    segments,
                    str(video),
                    active_start,
                    float(timestamp_seconds),
                )
                active_start = None

            if position == len(timestamps) - 1 and active_start is not None:
                append_segment(
                    segments,
                    str(video),
                    active_start,
                    float(timestamp_seconds) + final_frame_duration,
                )

    return segments


def smooth_segments(
    segments: list[dict[str, object]],
    merge_gap_seconds: float,
    min_duration_seconds: float,
) -> list[dict[str, object]]:
    if not segments:
        return []

    smoothed: list[dict[str, object]] = []
    for segment in segments:
        if (
            smoothed
            and segment["video"] == smoothed[-1]["video"]
            and float(segment["start_seconds"]) - float(smoothed[-1]["end_seconds"])
            <= merge_gap_seconds
        ):
            smoothed[-1]["end_seconds"] = max(
                float(smoothed[-1]["end_seconds"]),
                float(segment["end_seconds"]),
            )
            smoothed[-1]["end"] = seconds_to_timestamp(float(smoothed[-1]["end_seconds"]))
        else:
            smoothed.append(segment.copy())

    return [
        segment
        for segment in smoothed
        if float(segment["end_seconds"]) - float(segment["start_seconds"])
        >= min_duration_seconds
    ]


def set_segment_bounds(
    segment: dict[str, object],
    start_seconds: float,
    end_seconds: float,
) -> dict[str, object]:
    updated = segment.copy()
    updated.update(
        {
            "start": seconds_to_timestamp(start_seconds),
            "end": seconds_to_timestamp(end_seconds),
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
        }
    )
    return updated


def force_segment_count(
    segments: list[dict[str, object]],
    predictions: pd.DataFrame,
    expected_count: int,
    min_duration_seconds: float,
) -> list[dict[str, object]]:
    """Force a fixed count using raw-score valleys as transition boundaries."""
    if expected_count <= 0:
        raise ValueError("expected_count must be positive.")
    if min_duration_seconds <= 0:
        raise ValueError("min_duration_seconds must be positive when forcing segments.")

    score_column = next(
        (column for column in ("score", "smoothed_score") if column in predictions),
        None,
    )
    if score_column is None:
        raise ValueError("Predictions need score or smoothed_score to split segments.")

    ordered = predictions[["video", "timestamp_seconds", score_column]].copy()
    ordered["video"] = ordered["video"].astype(str)
    ordered["timestamp_seconds"] = pd.to_numeric(
        ordered["timestamp_seconds"], errors="raise"
    )
    ordered[score_column] = pd.to_numeric(ordered[score_column], errors="raise")
    if not np.isfinite(ordered[["timestamp_seconds", score_column]]).all().all():
        raise ValueError("Predictions contain non-finite timestamps or scores.")
    ordered = ordered.sort_values(["video", "timestamp_seconds"])
    ordered["_transition_score"] = ordered.groupby("video", sort=False)[
        score_column
    ].transform(lambda values: values.rolling(3, center=True, min_periods=1).mean())

    output: list[dict[str, object]] = []
    videos = ordered["video"].drop_duplicates().tolist()
    for video in videos:
        current = sorted(
            (segment.copy() for segment in segments if str(segment["video"]) == video),
            key=lambda segment: float(segment["start_seconds"]),
        )
        if not current:
            raise ValueError(
                f"Cannot force {expected_count} segments for {video}: none found."
            )

        video_scores = ordered[ordered["video"] == video]
        while len(current) < expected_count:
            choices: list[tuple[float, float, int]] = []
            for index, segment in enumerate(current):
                start = float(segment["start_seconds"]) + min_duration_seconds
                end = float(segment["end_seconds"]) - min_duration_seconds
                candidates = video_scores[
                    video_scores["timestamp_seconds"].between(start, end)
                ]
                if candidates.empty:
                    continue
                best = candidates.sort_values(
                    ["_transition_score", "timestamp_seconds"]
                ).iloc[0]
                choices.append(
                    (
                        float(best["_transition_score"]),
                        float(best["timestamp_seconds"]),
                        index,
                    )
                )
            if not choices:
                raise ValueError(
                    f"Cannot split {video} into {expected_count} segments while "
                    f"keeping each at least {min_duration_seconds:g}s."
                )

            _, split_seconds, index = min(choices)
            segment = current[index]
            current[index : index + 1] = [
                set_segment_bounds(
                    segment,
                    float(segment["start_seconds"]),
                    split_seconds,
                ),
                set_segment_bounds(
                    segment,
                    split_seconds,
                    float(segment["end_seconds"]),
                ),
            ]

        while len(current) > expected_count:
            index = min(
                range(len(current) - 1),
                key=lambda position: (
                    float(current[position + 1]["start_seconds"])
                    - float(current[position]["end_seconds"])
                ),
            )
            current[index : index + 2] = [
                set_segment_bounds(
                    current[index],
                    float(current[index]["start_seconds"]),
                    float(current[index + 1]["end_seconds"]),
                )
            ]
        output.extend(current)
    return output


def main() -> int:
    args = parse_args()
    if args.merge_gap_seconds < 0:
        print("--merge-gap-seconds cannot be negative.")
        return 1
    if args.min_duration_seconds < 0:
        print("--min-duration-seconds cannot be negative.")
        return 1
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        print("--threshold must be between 0 and 1.")
        return 1
    if (
        args.expected_segments_per_video is not None
        and args.expected_segments_per_video <= 0
    ):
        print("--expected-segments-per-video must be positive.")
        return 1
    if (
        args.expected_segments_per_video is not None
        and args.min_duration_seconds <= 0
    ):
        print(
            "--min-duration-seconds must be positive when forcing segment count."
        )
        return 1

    if not args.predictions_csv.exists():
        print(f"Predictions CSV does not exist: {args.predictions_csv}")
        print("Run python src/lineup/predict_mobilenet.py first.")
        return 1

    predictions = pd.read_csv(args.predictions_csv)
    try:
        segments = predictions_to_segments(predictions, threshold=args.threshold)
    except (TypeError, ValueError) as exc:
        print(f"Invalid predictions in {args.predictions_csv}: {exc}")
        return 1

    segments = smooth_segments(
        segments,
        merge_gap_seconds=args.merge_gap_seconds,
        min_duration_seconds=args.min_duration_seconds,
    )
    if args.expected_segments_per_video is not None:
        try:
            segments = force_segment_count(
                segments,
                predictions,
                expected_count=args.expected_segments_per_video,
                min_duration_seconds=args.min_duration_seconds,
            )
        except (TypeError, ValueError) as exc:
            print(f"Cannot force segment count: {exc}")
            return 1

    ensure_dir(args.output_csv.parent)
    pd.DataFrame(
        segments,
        columns=["video", "start", "end", "start_seconds", "end_seconds"],
    ).to_csv(args.output_csv, index=False)
    print(f"Saved segments to: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
