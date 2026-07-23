"""Extract detected lineup segments at 2 FPS and run PaddleOCR on the frames."""

from __future__ import annotations

import argparse
import math
import os
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENTS_CSV = (
    PROJECT_ROOT / "outputs" / "predictions" / "lineup_segments.csv"
)
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "data" / "raw_videos"
DEFAULT_FRAMES_DIR = PROJECT_ROOT / "data" / "ocr_frames"
DEFAULT_FRAMES_CSV = PROJECT_ROOT / "outputs" / "predictions" / "ocr_frames.csv"
DEFAULT_OUTPUT_CSV = (
    PROJECT_ROOT / "outputs" / "predictions" / "ocr_raw_detections.csv"
)
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache" / "paddlex"
DETECTION_MODEL = "PP-OCRv6_small_det"
RECOGNITION_MODEL = "PP-OCRv6_small_rec"

FRAME_COLUMNS = [
    "video",
    "segment_index",
    "segment_label",
    "segment_start_seconds",
    "segment_end_seconds",
    "frame_index",
    "frame_path",
    "timestamp",
    "timestamp_seconds",
    "relative_seconds",
    "frame_width",
    "frame_height",
]

DETECTION_COLUMNS = FRAME_COLUMNS + [
    "text",
    "text_type",
    "score",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "center_x_norm",
    "center_y_norm",
]


class LineupOCRError(Exception):
    """Raised when lineup-frame extraction or OCR cannot be completed."""


@dataclass(frozen=True)
class Segment:
    video: str
    video_path: Path
    index: int
    label: str
    start_seconds: float
    end_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read lineup segments, extract only those ranges at 2 FPS, and run "
            "PaddleOCR on the extracted frames."
        )
    )
    parser.add_argument("--segments-csv", type=Path, default=DEFAULT_SEGMENTS_CSV)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR)
    parser.add_argument("--frames-csv", type=Path, default=DEFAULT_FRAMES_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames per second inside detected lineup segments (default: 2).",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for OCR frames (default: 95).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.80,
        help="Discard OCR detections below this confidence (default: 0.80).",
    )
    parser.add_argument(
        "--ocr-batch-size",
        type=int,
        default=8,
        help="Number of frame paths submitted to PaddleOCR per call (default: 8).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Local PaddleX model cache.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Extract 2 FPS lineup frames and metadata without running OCR.",
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def relative_to_project(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def safe_stem(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_name).strip("._-")
    return safe_name or "video"


def seconds_to_timestamp(seconds: float) -> str:
    milliseconds = int(round(float(seconds) * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    base = f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}"
    return f"{base}.{milliseconds:03d}".rstrip("0") if milliseconds else base


def timestamp_to_seconds(value: object) -> float:
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = str(value).strip()
        try:
            seconds = float(text)
        except ValueError:
            parts = text.split(":")
            if len(parts) == 3:
                hours_text, minutes_text, seconds_text = parts
            elif len(parts) == 2:
                hours_text, minutes_text, seconds_text = "0", *parts
            else:
                raise ValueError(f"Invalid timestamp: {value}") from None
            seconds = (
                int(hours_text) * 3600
                + int(minutes_text) * 60
                + float(seconds_text)
            )
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"Invalid timestamp: {value}")
    return seconds


def resolve_video(video: str, video_dir: Path) -> Path:
    candidate = Path(video).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    from_video_dir = (video_dir / candidate).resolve()
    if from_video_dir.is_file():
        return from_video_dir

    from_project = (PROJECT_ROOT / candidate).resolve()
    return from_project if from_project.is_file() else from_video_dir


def load_segments(segments_csv: Path, video_dir: Path) -> list[Segment]:
    if not segments_csv.is_file():
        raise LineupOCRError(f"Segments CSV does not exist: {segments_csv}")

    rows = pd.read_csv(segments_csv)
    if rows.empty:
        raise LineupOCRError(f"Segments CSV has no rows: {segments_csv}")
    if "video" not in rows.columns:
        raise LineupOCRError("Segments CSV is missing the 'video' column.")

    has_seconds = {"start_seconds", "end_seconds"}.issubset(rows.columns)
    has_timestamps = {"start", "end"}.issubset(rows.columns)
    if not has_seconds and not has_timestamps:
        raise LineupOCRError(
            "Segments CSV must contain start_seconds/end_seconds or start/end."
        )

    start_column, end_column = (
        ("start_seconds", "end_seconds") if has_seconds else ("start", "end")
    )
    counters: defaultdict[str, int] = defaultdict(int)
    segments: list[Segment] = []

    for row_number, row in rows.iterrows():
        video = str(row["video"]).strip()
        if not video:
            raise LineupOCRError(f"Row {row_number + 2} has an empty video.")
        try:
            start_seconds = timestamp_to_seconds(row[start_column])
            end_seconds = timestamp_to_seconds(row[end_column])
        except (TypeError, ValueError) as exc:
            raise LineupOCRError(
                f"Invalid timestamp at row {row_number + 2}: {exc}"
            ) from exc
        if end_seconds <= start_seconds:
            raise LineupOCRError(
                f"Row {row_number + 2} must end after it starts."
            )

        video_path = resolve_video(video, video_dir)
        if not video_path.is_file():
            raise LineupOCRError(f"Source video does not exist: {video_path}")

        video_key = str(video_path)
        counters[video_key] += 1
        index = counters[video_key]
        raw_label = (
            str(row["segment"]).strip()
            if "segment" in rows.columns and pd.notna(row["segment"])
            else ""
        )
        segments.append(
            Segment(
                video=video,
                video_path=video_path,
                index=index,
                label=raw_label or f"lineup_{index:02d}",
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
        )

    return segments


def extract_segment_frames(
    segment: Segment,
    frames_dir: Path,
    fps: float,
    jpeg_quality: int,
) -> list[dict[str, object]]:
    capture = cv2.VideoCapture(str(segment.video_path))
    if not capture.isOpened():
        raise LineupOCRError(f"Cannot open video: {segment.video_path}")

    try:
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        source_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if source_fps <= 0 or source_frame_count <= 0:
            raise LineupOCRError(
                f"Cannot read FPS/frame count from: {segment.video_path}"
            )

        video_duration = source_frame_count / source_fps
        if segment.start_seconds >= video_duration:
            raise LineupOCRError(
                f"Segment starts outside video duration: {segment.video_path.name}"
            )
        extraction_end = min(segment.end_seconds, video_duration)

        output_dir = (
            frames_dir
            / safe_stem(segment.video_path.stem)
            / f"segment_{segment.index:02d}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        records: list[dict[str, object]] = []
        step_seconds = 1.0 / fps
        frame_index = 1
        timestamp_seconds = segment.start_seconds

        while timestamp_seconds < extraction_end:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000)
            ok, frame = capture.read()
            if not ok:
                raise LineupOCRError(
                    f"Cannot read {segment.video_path.name} at "
                    f"{timestamp_seconds:.3f}s"
                )

            timestamp_ms = round(timestamp_seconds * 1000)
            frame_path = output_dir / (
                f"frame_{frame_index:06d}_t{timestamp_ms:010d}.jpg"
            )
            if not cv2.imwrite(
                str(frame_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            ):
                raise LineupOCRError(f"Cannot write OCR frame: {frame_path}")

            frame_height, frame_width = frame.shape[:2]
            records.append(
                {
                    "video": segment.video,
                    "segment_index": segment.index,
                    "segment_label": segment.label,
                    "segment_start_seconds": segment.start_seconds,
                    "segment_end_seconds": segment.end_seconds,
                    "frame_index": frame_index,
                    "frame_path": relative_to_project(frame_path),
                    "timestamp": seconds_to_timestamp(timestamp_seconds),
                    "timestamp_seconds": round(timestamp_seconds, 3),
                    "relative_seconds": round(
                        timestamp_seconds - segment.start_seconds, 3
                    ),
                    "frame_width": frame_width,
                    "frame_height": frame_height,
                }
            )
            frame_index += 1
            timestamp_seconds = round(
                segment.start_seconds + (frame_index - 1) * step_seconds,
                6,
            )

        return records
    finally:
        capture.release()


def write_csv(
    rows: list[dict[str, object]],
    output_csv: Path,
    columns: list[str],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(output_csv, index=False)


def classify_text(text: str) -> str:
    if text.isdigit() and 1 <= int(text) <= 99:
        return "shirt_number_candidate"
    return "text"


def create_ocr(cache_dir: Path, recognition_batch_size: int):
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    try:
        from paddleocr import PaddleOCR
    except ModuleNotFoundError as exc:
        raise LineupOCRError(
            "PaddleOCR is not installed. Run this script with "
            "`.venv-ocr/bin/python`."
        ) from exc

    return PaddleOCR(
        text_detection_model_name=DETECTION_MODEL,
        text_recognition_model_name=RECOGNITION_MODEL,
        text_recognition_batch_size=recognition_batch_size,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
    )


def result_data(result: object) -> dict[str, object]:
    payload = getattr(result, "json", None)
    if callable(payload):
        payload = payload()
    if not isinstance(payload, dict):
        raise LineupOCRError("PaddleOCR returned an unsupported result format.")
    data = payload.get("res", payload)
    if not isinstance(data, dict):
        raise LineupOCRError("PaddleOCR result does not contain a result object.")
    return data


def detections_from_result(
    frame_record: dict[str, object],
    result: object,
    min_score: float,
) -> list[dict[str, object]]:
    data = result_data(result)
    texts = data.get("rec_texts", [])
    scores = data.get("rec_scores", [])
    boxes = data.get("rec_boxes", [])
    if not isinstance(texts, list) or len(texts) != len(scores) or len(texts) != len(boxes):
        raise LineupOCRError("PaddleOCR returned inconsistent detection arrays.")

    frame_width = int(frame_record["frame_width"])
    frame_height = int(frame_record["frame_height"])
    detections: list[dict[str, object]] = []

    for raw_text, raw_score, raw_box in zip(texts, scores, boxes, strict=True):
        text = str(raw_text).strip()
        score = float(raw_score)
        if not text or score < min_score:
            continue

        box = list(map(int, raw_box))
        if len(box) != 4:
            raise LineupOCRError(f"Unsupported OCR box: {raw_box}")
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        detections.append(
            {
                **frame_record,
                "text": text,
                "text_type": classify_text(text),
                "score": round(score, 6),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": round(center_x, 3),
                "center_y": round(center_y, 3),
                "center_x_norm": round(center_x / frame_width, 6),
                "center_y_norm": round(center_y / frame_height, 6),
            }
        )
    return detections


def run_ocr(
    frame_records: list[dict[str, object]],
    cache_dir: Path,
    output_csv: Path,
    min_score: float,
    batch_size: int,
) -> list[dict[str, object]]:
    print(f"Loading OCR models from: {cache_dir}")
    ocr = create_ocr(cache_dir, recognition_batch_size=batch_size)
    detections: list[dict[str, object]] = []

    for start in range(0, len(frame_records), batch_size):
        batch = frame_records[start : start + batch_size]
        input_paths = [
            str(resolve_project_path(Path(str(record["frame_path"]))))
            for record in batch
        ]
        results = list(ocr.predict(input_paths))
        if len(results) != len(batch):
            raise LineupOCRError(
                f"PaddleOCR returned {len(results)} result(s) for "
                f"{len(batch)} input frame(s)."
            )

        for frame_record, result in zip(batch, results, strict=True):
            detections.extend(
                detections_from_result(frame_record, result, min_score=min_score)
            )

        completed = min(start + len(batch), len(frame_records))
        write_csv(detections, output_csv, DETECTION_COLUMNS)
        print(
            f"OCR progress: {completed}/{len(frame_records)} frames, "
            f"{len(detections)} detections"
        )

    return detections


def validate_args(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise LineupOCRError("--fps must be greater than 0.")
    if not 1 <= args.jpeg_quality <= 100:
        raise LineupOCRError("--jpeg-quality must be between 1 and 100.")
    if not 0 <= args.min_score <= 1:
        raise LineupOCRError("--min-score must be between 0 and 1.")
    if args.ocr_batch_size <= 0:
        raise LineupOCRError("--ocr-batch-size must be greater than 0.")


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        segments_csv = resolve_project_path(args.segments_csv)
        video_dir = resolve_project_path(args.video_dir)
        frames_dir = resolve_project_path(args.frames_dir)
        frames_csv = resolve_project_path(args.frames_csv)
        output_csv = resolve_project_path(args.output_csv)
        cache_dir = resolve_project_path(args.cache_dir)
        segments = load_segments(segments_csv, video_dir)

        frame_records: list[dict[str, object]] = []
        for position, segment in enumerate(segments, start=1):
            print(
                f"[{position}/{len(segments)}] Extracting "
                f"{segment.video_path.name} segment {segment.index}: "
                f"{segment.start_seconds:.3f}-{segment.end_seconds:.3f}s "
                f"at {args.fps:g} FPS"
            )
            frame_records.extend(
                extract_segment_frames(
                    segment,
                    frames_dir=frames_dir,
                    fps=args.fps,
                    jpeg_quality=args.jpeg_quality,
                )
            )

        write_csv(frame_records, frames_csv, FRAME_COLUMNS)
        print(f"Extracted {len(frame_records)} lineup frames.")
        print(f"Frame metadata saved to: {frames_csv}")

        if args.extract_only:
            print("Extraction-only mode: OCR was not run.")
            return 0
        if not frame_records:
            raise LineupOCRError("No lineup frames were extracted.")

        detections = run_ocr(
            frame_records,
            cache_dir=cache_dir,
            output_csv=output_csv,
            min_score=args.min_score,
            batch_size=args.ocr_batch_size,
        )
        print(f"Saved {len(detections)} OCR detections to: {output_csv}")
        return 0
    except (LineupOCRError, OSError, pd.errors.ParserError) as exc:
        print(f"Lineup OCR failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
