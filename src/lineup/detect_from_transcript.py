"""CLI: detect starting-lineup intervals from a persisted transcript."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from lineup.detector import (
    DEFAULT_TEXT_MODEL,
    LineupDetectionError,
    QwenLineupDetector,
    write_lineup_csv,
)
from lineup.utils import PROJECT_ROOT, resolve_project_path
from transcript.qwen import DEFAULT_BASE_URL
from transcript.schema import TranscriptValidationError, read_jsonl, write_json


DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs" / "predictions" / "lineup" / "lineup_segments.csv"
)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect football starting-lineup intervals from transcript JSONL."
    )
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument(
        "--transcription-metadata",
        type=Path,
        help="Defaults to metadata.json beside the transcript.",
    )
    parser.add_argument(
        "--video",
        help="Video filename stored in the CSV; defaults to transcription metadata.",
    )
    parser.add_argument(
        "--model",
        help="Defaults to QWEN_TEXT_MODEL or qwen-plus.",
    )
    parser.add_argument(
        "--base-url",
        help="DashScope API base URL; defaults to the Singapore endpoint.",
    )
    parser.add_argument(
        "--expected-lineups",
        type=_expected_lineups,
        default=2,
        help=(
            "Expected coarse lineup count; default is 2. Use auto to accept "
            "0-2 validated results without enforcing completeness."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(PROJECT_ROOT / ".env")
    try:
        transcript_path = resolve_project_path(args.transcript)
        transcription_metadata_path = resolve_project_path(
            args.transcription_metadata
            if args.transcription_metadata is not None
            else transcript_path.parent / "metadata.json"
        )
        if not transcription_metadata_path.is_file():
            raise LineupDetectionError(
                f"Transcription metadata does not exist: {transcription_metadata_path}"
            )
        transcription_metadata = json.loads(
            transcription_metadata_path.read_text(encoding="utf-8")
        )
        source_video = str(transcription_metadata.get("source_video", "")).strip()
        video_name = args.video or Path(source_video).name
        if not video_name:
            raise LineupDetectionError(
                "Video name is missing; pass --video or provide source_video metadata."
            )
        window_start = float(
            transcription_metadata.get("source_start_seconds", 0.0)
        )
        window_end = float(
            transcription_metadata.get(
                "processed_end_seconds",
                transcription_metadata.get(
                    "requested_end_seconds",
                    window_start
                    + float(
                        transcription_metadata["requested_duration_seconds"]
                    ),
                ),
            )
        )

        output_path = resolve_project_path(args.output)
        raw_output_path = resolve_project_path(
            args.raw_output
            if args.raw_output is not None
            else output_path.parent / "qwen_detection_raw_response.json"
        )
        detection_metadata_path = resolve_project_path(
            args.metadata_output
            if args.metadata_output is not None
            else output_path.parent / "detection_metadata.json"
        )
        for output in (output_path, raw_output_path, detection_metadata_path):
            if output.exists() and not args.overwrite:
                raise FileExistsError(
                    f"Output already exists; pass --overwrite: {output}"
                )

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise LineupDetectionError(
                "Missing API key. Set DASHSCOPE_API_KEY in the project .env file."
            )
        model = args.model or os.getenv("QWEN_TEXT_MODEL") or DEFAULT_TEXT_MODEL
        base_url = (
            args.base_url or os.getenv("DASHSCOPE_BASE_URL") or DEFAULT_BASE_URL
        )
        transcript = read_jsonl(transcript_path)
        print(
            f"Detecting lineup in {len(transcript)} transcript segment(s), "
            f"window {window_start:.3f}-{window_end:.3f}s"
        )
        result = QwenLineupDetector(
            api_key=api_key,
            model=model,
            base_url=base_url,
            expected_segment_count=args.expected_lineups,
        ).detect(
            transcript,
            video_name=video_name,
            window_start_seconds=window_start,
            window_end_seconds=window_end,
        )
        write_lineup_csv(result, video_name=video_name, output_path=output_path)
        write_json(result.raw_response, raw_output_path)
        write_json(
            {
                "provider": "qwen",
                "model": result.model,
                "video": video_name,
                "source_transcript": str(transcript_path),
                "transcription_metadata": str(transcription_metadata_path),
                "window_start_seconds": window_start,
                "window_end_seconds": window_end,
                "transcript_segment_count": len(transcript),
                "lineup_segment_count": len(result.segments),
                "expected_lineup_segment_count": args.expected_lineups,
                "detection_status": result.status,
                "requires_review": (
                    args.expected_lineups is not None
                    and len(result.segments) != args.expected_lineups
                ),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            detection_metadata_path,
        )
        print(f"Detected lineup segments: {len(result.segments)}")
        print(f"Detection status: {result.status}")
        print(f"Lineup CSV: {output_path}")
        if (
            args.expected_lineups is not None
            and len(result.segments) != args.expected_lineups
        ):
            print(
                "Detection is incomplete and requires manual review.",
                file=sys.stderr,
            )
            return 2
        return 0
    except (
        FileExistsError,
        json.JSONDecodeError,
        KeyError,
        LineupDetectionError,
        OSError,
        TranscriptValidationError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Lineup detection failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
