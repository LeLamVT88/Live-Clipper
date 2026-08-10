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
    GeminiLineupDetector,
    LineupDetectionError,
    write_lineup_csv,
)
from transcript.gemini import DEFAULT_MODEL
from transcript.schema import TranscriptValidationError, read_jsonl, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs" / "predictions" / "lineup" / "lineup_segments.csv"
)


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
    parser.add_argument("--model", help="Defaults to GEMINI_MODEL.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path.expanduser().resolve()
    return (PROJECT_ROOT / path).resolve()


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
                "requested_end_seconds",
                window_start
                + float(transcription_metadata["requested_duration_seconds"]),
            )
        )

        output_path = resolve_project_path(args.output)
        raw_output_path = resolve_project_path(
            args.raw_output
            if args.raw_output is not None
            else output_path.parent / "gemini_detection_raw_response.json"
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

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise LineupDetectionError(
                "Missing API key. Set GEMINI_API_KEY in the project .env file."
            )
        model = args.model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL
        transcript = read_jsonl(transcript_path)
        print(
            f"Detecting lineup in {len(transcript)} transcript segment(s), "
            f"window {window_start:.3f}-{window_end:.3f}s"
        )
        result = GeminiLineupDetector(api_key=api_key, model=model).detect(
            transcript,
            video_name=video_name,
            window_start_seconds=window_start,
            window_end_seconds=window_end,
        )
        write_lineup_csv(result, video_name=video_name, output_path=output_path)
        write_json(result.raw_response, raw_output_path)
        write_json(
            {
                "provider": "gemini",
                "model": result.model,
                "video": video_name,
                "source_transcript": str(transcript_path),
                "transcription_metadata": str(transcription_metadata_path),
                "window_start_seconds": window_start,
                "window_end_seconds": window_end,
                "transcript_segment_count": len(transcript),
                "lineup_segment_count": len(result.segments),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "ground_truth_used": False,
            },
            detection_metadata_path,
        )
        print(f"Detected lineup segments: {len(result.segments)}")
        print(f"Lineup CSV: {output_path}")
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
