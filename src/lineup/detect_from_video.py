"""CLI: detect football lineup graphics directly from video frames."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lineup.media import MediaProbeError, probe_media
from lineup.scene_detection import (
    DEFAULT_SCENE_MIN_LENGTH_SECONDS,
    DEFAULT_SCENE_SNAP_RADIUS_SECONDS,
    DEFAULT_SCENE_THRESHOLD,
)
from lineup.schema import LineupDetectionError, write_lineup_csv
from lineup.utils import (
    default_run_dir,
    resolve_project_path,
    timestamp_to_seconds,
    write_json,
)
from lineup.video_detection import (
    DEFAULT_COARSE_UNIT_SECONDS,
    DEFAULT_DENSE_PADDING_SECONDS,
    DEFAULT_DENSE_UNIT_SECONDS,
    DEFAULT_MAX_COARSE_CANDIDATES,
    DEFAULT_SAFETY_PADDING_AFTER_SECONDS,
    DEFAULT_SAFETY_PADDING_BEFORE_SECONDS,
    DEFAULT_VISUAL_MODEL,
    detect_lineups_from_video,
)
from lineup.visual_scan import DEFAULT_OCR_PYTHON


DEFAULT_SCAN_END_SECONDS = 600.0


def _expected_lineups(value: str) -> int | None:
    normalized = value.strip().casefold()
    if normalized in {"auto", "none", "0"}:
        return None
    try:
        count = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected lineups must be 1, 2, or auto"
        ) from exc
    if count not in {1, 2}:
        raise argparse.ArgumentTypeError(
            "expected lineups must be 1, 2, or auto"
        )
    return count


def _scan_end(value: str) -> float | None:
    if value.strip().casefold() in {"full", "end", "all"}:
        return None
    try:
        return timestamp_to_seconds(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _timestamp(value: str) -> float:
    try:
        return timestamp_to_seconds(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect football starting-lineup graphics directly from video "
            "with PySceneDetect and two-pass tiny OCR; no transcript required."
        )
    )
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument(
        "--video",
        help="Video filename stored in the CSV; defaults to --source-video name.",
    )
    parser.add_argument(
        "--scan-start",
        type=_timestamp,
        default=0.0,
        help="Scan start in seconds or HH:MM:SS; default 0.",
    )
    parser.add_argument(
        "--scan-end",
        type=_scan_end,
        default=DEFAULT_SCAN_END_SECONDS,
        help=(
            "Scan end in seconds or HH:MM:SS; default 600. Use 'full' to scan "
            "the complete video."
        ),
    )
    parser.add_argument(
        "--expected-lineups",
        type=_expected_lineups,
        default=2,
        help="Expected lineup count; default 2. Use auto for 0-2.",
    )
    parser.add_argument(
        "--coarse-unit-seconds",
        type=float,
        default=DEFAULT_COARSE_UNIT_SECONDS,
        help="Maximum gap between coarse representative frames; default 3s.",
    )
    parser.add_argument(
        "--dense-unit-seconds",
        type=float,
        default=DEFAULT_DENSE_UNIT_SECONDS,
        help="Candidate rescan interval; default 0.5s.",
    )
    parser.add_argument(
        "--dense-padding",
        type=float,
        default=DEFAULT_DENSE_PADDING_SECONDS,
        help="Seconds rescanned around each coarse candidate; default 12s.",
    )
    parser.add_argument(
        "--padding-before",
        type=float,
        default=DEFAULT_SAFETY_PADDING_BEFORE_SECONDS,
        help="Conservative padding before the visual boundary; default 4s.",
    )
    parser.add_argument(
        "--padding-after",
        type=float,
        default=DEFAULT_SAFETY_PADDING_AFTER_SECONDS,
        help="Conservative padding after the visual boundary; default 6s.",
    )
    parser.add_argument(
        "--max-coarse-candidates",
        type=int,
        default=DEFAULT_MAX_COARSE_CANDIDATES,
        help="Candidate budget before dense rescanning; default 6.",
    )
    parser.add_argument(
        "--ocr-python",
        type=Path,
        default=DEFAULT_OCR_PYTHON,
        help="Python executable containing PaddleOCR; default .venv-ocr.",
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=DEFAULT_SCENE_THRESHOLD,
    )
    parser.add_argument("--scene-min-length-frames", type=int)
    parser.add_argument(
        "--scene-min-length-seconds",
        type=float,
        default=DEFAULT_SCENE_MIN_LENGTH_SECONDS,
    )
    parser.add_argument(
        "--scene-snap-radius",
        type=float,
        default=DEFAULT_SCENE_SNAP_RADIUS_SECONDS,
    )
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        source = resolve_project_path(args.source_video)
        media = probe_media(source, ffprobe=args.ffprobe)
        if not media.has_video:
            raise LineupDetectionError(f"Input has no video stream: {source}")
        scan_start = float(args.scan_start)
        requested_end = (
            media.duration_seconds if args.scan_end is None else float(args.scan_end)
        )
        scan_end = min(media.duration_seconds, requested_end)
        if scan_start >= scan_end:
            raise LineupDetectionError(
                "The requested scan range does not overlap the source video."
            )

        output_dir = default_run_dir(source) / "predictions" / "lineup"
        output_path = resolve_project_path(
            args.output or output_dir / "lineup_segments.csv"
        )
        raw_output_path = resolve_project_path(
            args.raw_output
            or output_path.parent / "visual_detection_raw_response.json"
        )
        metadata_output_path = resolve_project_path(
            args.metadata_output or output_path.parent / "detection_metadata.json"
        )
        for output in (output_path, raw_output_path, metadata_output_path):
            if output.exists() and not args.overwrite:
                raise FileExistsError(
                    f"Output already exists; pass --overwrite: {output}"
                )

        print(
            "Scanning scene representatives and OCR text layouts in "
            f"{scan_start:.3f}-{scan_end:.3f}s"
        )
        result = detect_lineups_from_video(
            source,
            scan_start_seconds=scan_start,
            scan_end_seconds=scan_end,
            expected_count=args.expected_lineups,
            scene_threshold=args.scene_threshold,
            scene_min_length_frames=args.scene_min_length_frames,
            scene_min_length_seconds=args.scene_min_length_seconds,
            scene_snap_radius_seconds=args.scene_snap_radius,
            coarse_unit_seconds=args.coarse_unit_seconds,
            dense_unit_seconds=args.dense_unit_seconds,
            dense_padding_seconds=args.dense_padding,
            safety_padding_before_seconds=args.padding_before,
            safety_padding_after_seconds=args.padding_after,
            max_coarse_candidates=args.max_coarse_candidates,
            ocr_python=args.ocr_python,
        )
        review_reasons = result.raw_response.get("_review_reasons", [])
        if not isinstance(review_reasons, list):
            review_reasons = []
        requires_review = bool(review_reasons) or (
            args.expected_lineups is not None
            and len(result.segments) != args.expected_lineups
        )
        write_lineup_csv(
            result,
            video_name=args.video or source.name,
            output_path=output_path,
        )
        write_json(result.raw_response, raw_output_path)
        write_json(
            {
                "provider": "local",
                "model": DEFAULT_VISUAL_MODEL,
                "method": "scene_representative_ocr_then_dense_visual_scan",
                "uses_transcript": False,
                "uses_text_llm": False,
                "uses_visual_llm": False,
                "uses_ocr": True,
                "uses_pyscenedetect": True,
                "source_video": str(source),
                "source_duration_seconds": round(media.duration_seconds, 3),
                "scan_start_seconds": round(scan_start, 3),
                "scan_end_seconds": round(scan_end, 3),
                "lineup_segment_count": len(result.segments),
                "expected_lineup_segment_count": args.expected_lineups,
                "detection_status": result.status,
                "requires_review": requires_review,
                "review_reasons": review_reasons,
                "timestamp_precision": (
                    "dense_half_second_visual_evidence_with_outward_scene_snap_"
                    "and_safety_padding"
                ),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            metadata_output_path,
        )
        print(f"Detected lineup segments: {len(result.segments)}")
        print(f"Detection status: {result.status}")
        print(f"Lineup CSV: {output_path}")
        if requires_review:
            print(
                "Detection requires review: " + ", ".join(review_reasons),
                file=sys.stderr,
            )
        if (
            args.expected_lineups is not None
            and len(result.segments) != args.expected_lineups
        ):
            return 2
        return 0
    except (
        MediaProbeError,
        FileExistsError,
        json.JSONDecodeError,
        LineupDetectionError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Video lineup detection failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
