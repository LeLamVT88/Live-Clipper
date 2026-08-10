from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd

from utils import (
    ensure_dir,
    seconds_to_timestamp,
    timestamp_range_to_seconds,
    timestamp_to_seconds,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions" / "lineup"
DEFAULT_GROUND_TRUTH_CSV = PROJECT_ROOT / "data" / "ground_truth.csv"
DEFAULT_SEGMENTS_CSV = DEFAULT_OUTPUT_DIR / "lineup_segments.csv"
DEFAULT_DETAILS_OUTPUT = DEFAULT_OUTPUT_DIR / "segment_matches.csv"
DEFAULT_PER_VIDEO_OUTPUT = DEFAULT_OUTPUT_DIR / "segment_metrics_by_video.csv"
DEFAULT_METRICS_OUTPUT = DEFAULT_OUTPUT_DIR / "segment_metrics.csv"
GROUND_TRUTH_COLUMNS = {"video", "Đội 1", "Đội 2"}
TEAM_COLUMNS = ("Đội 1", "Đội 2")


class SegmentEvaluationError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate detected lineup time segments against ground truth."
    )
    parser.add_argument("--ground-truth-csv", type=Path, default=DEFAULT_GROUND_TRUTH_CSV)
    parser.add_argument("--segments-csv", type=Path, default=DEFAULT_SEGMENTS_CSV)
    parser.add_argument(
        "--video",
        action="append",
        default=[],
        help=(
            "Ground-truth video name to evaluate. Repeat for multiple videos; "
            "omit to evaluate every ground-truth video."
        ),
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="Minimum temporal IoU for a predicted segment to count as detected.",
    )
    parser.add_argument("--details-output", type=Path, default=DEFAULT_DETAILS_OUTPUT)
    parser.add_argument(
        "--per-video-output", type=Path, default=DEFAULT_PER_VIDEO_OUTPUT
    )
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_OUTPUT)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0 <= args.iou_threshold <= 1:
        raise SegmentEvaluationError("--iou-threshold must be between 0 and 1.")


def load_ground_truth(csv_path: Path) -> dict[str, list[dict[str, Any]]]:
    if not csv_path.exists():
        raise SegmentEvaluationError(f"Ground truth CSV does not exist: {csv_path}")

    frame = pd.read_csv(csv_path)
    missing = sorted(GROUND_TRUTH_COLUMNS - set(frame.columns))
    if missing:
        raise SegmentEvaluationError(
            f"Ground truth is missing columns: {', '.join(missing)}"
        )
    duplicated = frame["video"].duplicated(keep=False)
    if duplicated.any():
        videos = sorted(frame.loc[duplicated, "video"].astype(str).unique())
        raise SegmentEvaluationError(
            "Ground truth contains duplicate video rows: " + ", ".join(videos)
        )

    ranges_by_video: dict[str, list[dict[str, Any]]] = {}
    for row_number, row in frame.iterrows():
        video = str(row["video"]).strip()
        if not video:
            raise SegmentEvaluationError(
                f"Ground truth row {row_number + 2} has an empty video name."
            )

        ranges: list[dict[str, Any]] = []
        for team_column in TEAM_COLUMNS:
            try:
                start_seconds, end_seconds = timestamp_range_to_seconds(
                    row[team_column]
                )
            except ValueError as exc:
                raise SegmentEvaluationError(
                    f"Invalid {team_column} range for {video}: {exc}"
                ) from exc
            ranges.append(
                {
                    "team": team_column,
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                }
            )

        ranges.sort(key=lambda item: float(item["start_seconds"]))
        for previous, current in zip(ranges, ranges[1:]):
            if float(current["start_seconds"]) < float(previous["end_seconds"]):
                raise SegmentEvaluationError(
                    f"Ground truth lineup ranges overlap for {video}."
                )
        ranges_by_video[video] = ranges

    return ranges_by_video


def load_predicted_segments(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise SegmentEvaluationError(f"Segments CSV does not exist: {csv_path}")

    segments = pd.read_csv(csv_path)
    if "video" not in segments.columns:
        raise SegmentEvaluationError("Segments CSV is missing the video column.")

    has_seconds = {"start_seconds", "end_seconds"}.issubset(segments.columns)
    has_timestamps = {"start", "end"}.issubset(segments.columns)
    if not has_seconds and not has_timestamps:
        raise SegmentEvaluationError(
            "Segments CSV must contain start_seconds/end_seconds or start/end."
        )

    parsed = segments.copy()
    start_column, end_column = (
        ("start_seconds", "end_seconds") if has_seconds else ("start", "end")
    )
    parsed["video"] = parsed["video"].astype(str).str.strip()
    if (parsed["video"] == "").any():
        raise SegmentEvaluationError("Segments CSV contains an empty video name.")
    parsed["start_seconds"] = parsed[start_column].map(timestamp_to_seconds)
    parsed["end_seconds"] = parsed[end_column].map(timestamp_to_seconds)
    if (parsed["end_seconds"] <= parsed["start_seconds"]).any():
        raise SegmentEvaluationError("Every segment must end after it starts.")
    parsed["start"] = parsed["start_seconds"].map(seconds_to_timestamp)
    parsed["end"] = parsed["end_seconds"].map(seconds_to_timestamp)
    return parsed.sort_values(["video", "start_seconds"]).reset_index(drop=True)


def intersection_seconds(
    first_start: float,
    first_end: float,
    second_start: float,
    second_end: float,
) -> float:
    return max(0.0, min(first_end, second_end) - max(first_start, second_start))


def temporal_iou(ground_truth: dict[str, Any], prediction: dict[str, Any]) -> float:
    intersection = intersection_seconds(
        float(ground_truth["start_seconds"]),
        float(ground_truth["end_seconds"]),
        float(prediction["start_seconds"]),
        float(prediction["end_seconds"]),
    )
    if intersection == 0:
        return 0.0
    union = (
        float(ground_truth["end_seconds"])
        - float(ground_truth["start_seconds"])
        + float(prediction["end_seconds"])
        - float(prediction["start_seconds"])
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def ordered_best_matching(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[int, int]:
    """Find the maximum-total-IoU order-preserving one-to-one matching."""
    rows = len(ground_truth)
    columns = len(predictions)
    states: list[list[tuple[float, int, tuple[tuple[int, int], ...]]]] = [
        [(0.0, 0, tuple()) for _ in range(columns + 1)]
        for _ in range(rows + 1)
    ]

    def better(
        first: tuple[float, int, tuple[tuple[int, int], ...]],
        second: tuple[float, int, tuple[tuple[int, int], ...]],
    ) -> tuple[float, int, tuple[tuple[int, int], ...]]:
        first_key = (round(first[0], 12), first[1])
        second_key = (round(second[0], 12), second[1])
        return first if first_key >= second_key else second

    for ground_truth_count in range(1, rows + 1):
        for prediction_count in range(1, columns + 1):
            best = better(
                states[ground_truth_count - 1][prediction_count],
                states[ground_truth_count][prediction_count - 1],
            )
            iou = temporal_iou(
                ground_truth[ground_truth_count - 1],
                predictions[prediction_count - 1],
            )
            if iou > 0:
                previous = states[ground_truth_count - 1][prediction_count - 1]
                paired = (
                    previous[0] + iou,
                    previous[1] + 1,
                    previous[2]
                    + ((ground_truth_count - 1, prediction_count - 1),),
                )
                best = better(best, paired)
            states[ground_truth_count][prediction_count] = best

    return dict(states[rows][columns][2])


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def f1_score(precision: float, recall: float) -> float:
    return safe_divide(2 * precision * recall, precision + recall)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def evaluate_video(
    video: str,
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    split_name: str,
    iou_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matching = ordered_best_matching(ground_truth, predictions)
    matched_prediction_indices = set(matching.values())
    detail_rows: list[dict[str, Any]] = []
    detected_ious: list[float] = []
    all_ground_truth_ious: list[float] = []
    start_errors: list[float] = []
    end_errors: list[float] = []
    detected_segments = 0

    for ground_truth_index, truth in enumerate(ground_truth):
        prediction_index = matching.get(ground_truth_index)
        prediction = (
            predictions[prediction_index]
            if prediction_index is not None
            else None
        )
        iou = temporal_iou(truth, prediction) if prediction is not None else 0.0
        detected = prediction is not None and iou >= iou_threshold
        status = (
            "detected"
            if detected
            else "below_iou_threshold"
            if prediction is not None
            else "missed"
        )
        all_ground_truth_ious.append(iou)

        start_error = math.nan
        end_error = math.nan
        if prediction is not None:
            start_error = float(prediction["start_seconds"]) - float(
                truth["start_seconds"]
            )
            end_error = float(prediction["end_seconds"]) - float(
                truth["end_seconds"]
            )
        if detected:
            detected_segments += 1
            detected_ious.append(iou)
            start_errors.append(start_error)
            end_errors.append(end_error)

        detail_rows.append(
            {
                "split": split_name,
                "video": video,
                "ground_truth_segment": truth["team"],
                "status": status,
                "detected": detected,
                "ground_truth_start": seconds_to_timestamp(
                    float(truth["start_seconds"])
                ),
                "ground_truth_end": seconds_to_timestamp(
                    float(truth["end_seconds"])
                ),
                "ground_truth_start_seconds": truth["start_seconds"],
                "ground_truth_end_seconds": truth["end_seconds"],
                "prediction_index": (
                    prediction_index + 1 if prediction_index is not None else pd.NA
                ),
                "prediction_start": (
                    prediction["start"] if prediction is not None else pd.NA
                ),
                "prediction_end": (
                    prediction["end"] if prediction is not None else pd.NA
                ),
                "prediction_start_seconds": (
                    prediction["start_seconds"] if prediction is not None else math.nan
                ),
                "prediction_end_seconds": (
                    prediction["end_seconds"] if prediction is not None else math.nan
                ),
                "start_error_seconds": start_error,
                "end_error_seconds": end_error,
                "temporal_iou": iou,
            }
        )

    for prediction_index, prediction in enumerate(predictions):
        if prediction_index in matched_prediction_indices:
            continue
        detail_rows.append(
            {
                "split": split_name,
                "video": video,
                "ground_truth_segment": pd.NA,
                "status": "extra_prediction",
                "detected": False,
                "ground_truth_start": pd.NA,
                "ground_truth_end": pd.NA,
                "ground_truth_start_seconds": math.nan,
                "ground_truth_end_seconds": math.nan,
                "prediction_index": prediction_index + 1,
                "prediction_start": prediction["start"],
                "prediction_end": prediction["end"],
                "prediction_start_seconds": prediction["start_seconds"],
                "prediction_end_seconds": prediction["end_seconds"],
                "start_error_seconds": math.nan,
                "end_error_seconds": math.nan,
                "temporal_iou": 0.0,
            }
        )

    ground_truth_segments = len(ground_truth)
    predicted_segments = len(predictions)
    false_positive_segments = predicted_segments - detected_segments
    missed_segments = ground_truth_segments - detected_segments
    precision = safe_divide(detected_segments, predicted_segments)
    recall = safe_divide(detected_segments, ground_truth_segments)

    gap_duration = 0.0
    gap_false_positive_seconds = 0.0
    for first, second in zip(ground_truth, ground_truth[1:]):
        gap_start = float(first["end_seconds"])
        gap_end = float(second["start_seconds"])
        gap_duration += max(0.0, gap_end - gap_start)
        gap_false_positive_seconds += sum(
            intersection_seconds(
                gap_start,
                gap_end,
                float(prediction["start_seconds"]),
                float(prediction["end_seconds"]),
            )
            for prediction in predictions
        )
    gap_preservation = (
        1.0 - safe_divide(gap_false_positive_seconds, gap_duration)
        if gap_duration > 0
        else math.nan
    )

    metrics = {
        "split": split_name,
        "video": video,
        "ground_truth_segments": ground_truth_segments,
        "predicted_segments": predicted_segments,
        "detected_segments": detected_segments,
        "false_positive_segments": false_positive_segments,
        "missed_segments": missed_segments,
        "segment_precision": precision,
        "segment_recall": recall,
        "segment_f1": f1_score(precision, recall),
        "mean_iou_all_ground_truth": mean(all_ground_truth_ious),
        "mean_iou_detected": mean(detected_ious),
        "start_mae_seconds": mean([abs(value) for value in start_errors]),
        "end_mae_seconds": mean([abs(value) for value in end_errors]),
        "gap_duration_seconds": gap_duration,
        "gap_false_positive_seconds": gap_false_positive_seconds,
        "gap_preservation": gap_preservation,
        "all_ground_truth_detected": detected_segments == ground_truth_segments,
        "exact_detection": (
            detected_segments == ground_truth_segments
            and false_positive_segments == 0
        ),
    }
    return detail_rows, metrics


def summarize(
    per_video: pd.DataFrame,
    details: pd.DataFrame,
    split_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    ground_truth_rows = details[details["ground_truth_segment"].notna()]
    detected_rows = ground_truth_rows[ground_truth_rows["detected"]]
    ground_truth_segments = int(per_video["ground_truth_segments"].sum())
    predicted_segments = int(per_video["predicted_segments"].sum())
    detected_segments = int(per_video["detected_segments"].sum())
    false_positive_segments = predicted_segments - detected_segments
    missed_segments = ground_truth_segments - detected_segments
    precision = safe_divide(detected_segments, predicted_segments)
    recall = safe_divide(detected_segments, ground_truth_segments)
    gap_duration = float(per_video["gap_duration_seconds"].sum())
    gap_false_positive = float(per_video["gap_false_positive_seconds"].sum())

    return {
        "split": split_name,
        "videos": len(per_video),
        "iou_threshold": args.iou_threshold,
        "ground_truth_segments": ground_truth_segments,
        "predicted_segments": predicted_segments,
        "detected_segments": detected_segments,
        "false_positive_segments": false_positive_segments,
        "missed_segments": missed_segments,
        "segment_precision": precision,
        "segment_recall": recall,
        "segment_f1": f1_score(precision, recall),
        "mean_iou_all_ground_truth": float(
            ground_truth_rows["temporal_iou"].mean()
        ),
        "mean_iou_detected": float(detected_rows["temporal_iou"].mean()),
        "start_mae_seconds": float(
            detected_rows["start_error_seconds"].abs().mean()
        ),
        "end_mae_seconds": float(
            detected_rows["end_error_seconds"].abs().mean()
        ),
        "gap_duration_seconds": gap_duration,
        "gap_false_positive_seconds": gap_false_positive,
        "gap_preservation": (
            1.0 - safe_divide(gap_false_positive, gap_duration)
            if gap_duration > 0
            else math.nan
        ),
        "videos_all_ground_truth_detected": int(
            per_video["all_ground_truth_detected"].sum()
        ),
        "videos_exact_detection": int(per_video["exact_detection"].sum()),
    }


def print_summary(metrics: dict[str, Any]) -> None:
    print(
        f"Segment evaluation: split={metrics['split']} "
        f"IoU threshold={float(metrics['iou_threshold']):.2f}"
    )
    print(
        f"Videos={metrics['videos']} ground_truth={metrics['ground_truth_segments']} "
        f"predicted={metrics['predicted_segments']} "
        f"detected={metrics['detected_segments']} "
        f"false_positive={metrics['false_positive_segments']} "
        f"missed={metrics['missed_segments']}"
    )
    print(
        f"Segment precision={float(metrics['segment_precision']):.4f} "
        f"recall={float(metrics['segment_recall']):.4f} "
        f"F1={float(metrics['segment_f1']):.4f}"
    )
    print(
        f"Mean IoU={float(metrics['mean_iou_all_ground_truth']):.4f} "
        f"start MAE={float(metrics['start_mae_seconds']):.2f}s "
        f"end MAE={float(metrics['end_mae_seconds']):.2f}s"
    )
    print(
        f"Referee/handshake gap preservation="
        f"{float(metrics['gap_preservation']):.4f} "
        f"({float(metrics['gap_false_positive_seconds']):.2f}s predicted positive "
        f"of {float(metrics['gap_duration_seconds']):.2f}s)"
    )
    print(
        f"Videos with both lineups detected="
        f"{metrics['videos_all_ground_truth_detected']}/{metrics['videos']}; "
        f"exact detection with no extra segment="
        f"{metrics['videos_exact_detection']}/{metrics['videos']}"
    )


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        ground_truth_by_video = load_ground_truth(args.ground_truth_csv)
        segments = load_predicted_segments(args.segments_csv)

        prediction_videos = set(segments["video"].astype(str))
        missing_ground_truth = sorted(prediction_videos - set(ground_truth_by_video))
        if missing_ground_truth:
            raise SegmentEvaluationError(
                "Predicted videos are missing from ground truth: "
                + ", ".join(missing_ground_truth)
            )

        requested_videos = list(dict.fromkeys(args.video))
        unknown_videos = sorted(set(requested_videos) - set(ground_truth_by_video))
        if unknown_videos:
            raise SegmentEvaluationError(
                "Requested videos are missing from ground truth: "
                + ", ".join(unknown_videos)
            )
        evaluation_videos = requested_videos or sorted(ground_truth_by_video)
        split_name = "selected" if requested_videos else "all"

        segments_by_video: dict[str, list[dict[str, Any]]] = {
            video: [] for video in evaluation_videos
        }
        for segment in segments.to_dict("records"):
            video = str(segment["video"])
            if video in segments_by_video:
                segments_by_video[video].append(segment)

        detail_rows: list[dict[str, Any]] = []
        per_video_rows: list[dict[str, Any]] = []
        for video in evaluation_videos:
            video_details, video_metrics = evaluate_video(
                video=video,
                ground_truth=ground_truth_by_video[video],
                predictions=segments_by_video[video],
                split_name=split_name,
                iou_threshold=args.iou_threshold,
            )
            detail_rows.extend(video_details)
            per_video_rows.append(video_metrics)

        details = pd.DataFrame(detail_rows)
        per_video = pd.DataFrame(per_video_rows)
        metrics = summarize(per_video, details, split_name, args)

        for path in (
            args.details_output,
            args.per_video_output,
            args.metrics_output,
        ):
            ensure_dir(path.parent)
        details.to_csv(args.details_output, index=False)
        per_video.to_csv(args.per_video_output, index=False)
        pd.DataFrame([metrics]).to_csv(args.metrics_output, index=False)
    except (KeyError, TypeError, ValueError, pd.errors.ParserError) as exc:
        print(f"Segment evaluation failed: {exc}")
        return 1

    print_summary(metrics)
    print(f"Saved segment matches: {args.details_output}")
    print(f"Saved per-video metrics: {args.per_video_output}")
    print(f"Saved summary metrics: {args.metrics_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
