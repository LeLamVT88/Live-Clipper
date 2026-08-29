"""CLI: detect lineups from coarse transcript, then align to scene graphics."""

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
    DEFAULT_MAX_COARSE_LINEUP_DURATION_SECONDS,
    DEFAULT_TEXT_BASE_URL,
    DEFAULT_TEXT_MODEL,
    LineupDetectionError,
    QwenLineupDetector,
    QwenLineupValidationError,
    write_lineup_csv,
)
from lineup.scene_detection import (
    DEFAULT_MAX_AUDIO_END_LEAD_SECONDS,
    DEFAULT_MAX_GRAPHIC_DURATION_SECONDS,
    DEFAULT_SCENE_MIN_LENGTH_FRAMES,
    DEFAULT_SCENE_SNAP_RADIUS_SECONDS,
    DEFAULT_SCENE_THRESHOLD,
    detect_scene_cuts,
    snap_lineup_result,
)
from lineup.utils import (
    PROJECT_ROOT,
    default_lineup_prediction_dir,
    resolve_project_path,
)
from transcript.schema import TranscriptValidationError, read_jsonl, write_json


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
        description=(
            "Detect football starting-lineup intervals with one text-only "
            "Qwen call over the coarse transcript, then PySceneDetect."
        )
    )
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument(
        "--transcription-metadata",
        type=Path,
        help="Defaults to metadata.json beside the transcript.",
    )
    parser.add_argument(
        "--source-video",
        type=Path,
        help="Source used by PySceneDetect; defaults to metadata.source_video.",
    )
    parser.add_argument(
        "--video",
        help="Video filename stored in the CSV; defaults to transcription metadata.",
    )
    parser.add_argument(
        "--model",
        help=(
            "Transcript reasoning model; defaults to QWEN_LINEUP_MODEL or "
            "QWEN_TEXT_MODEL or qwen-plus."
        ),
    )
    parser.add_argument(
        "--base-url",
        help=(
            "OpenAI-compatible lineup model URL; defaults to "
            "QWEN_LINEUP_BASE_URL or the DashScope Singapore endpoint."
        ),
    )
    parser.add_argument(
        "--expected-lineups",
        type=_expected_lineups,
        default=2,
        help=(
            "Expected coarse lineup count; default is 2. Use auto to accept 0-2 "
            "validated results without enforcing completeness."
        ),
    )
    parser.add_argument(
        "--max-coarse-duration",
        type=float,
        default=DEFAULT_MAX_COARSE_LINEUP_DURATION_SECONDS,
        help=(
            "Reject and regenerate a Qwen lineup block longer than this many "
            "seconds; default is 75."
        ),
    )
    parser.add_argument(
        "--no-scene-snap",
        action="store_false",
        dest="scene_snap",
        help="Keep Qwen coarse boundaries without PySceneDetect alignment.",
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=DEFAULT_SCENE_THRESHOLD,
        help="PySceneDetect ContentDetector threshold; default is 27.",
    )
    parser.add_argument(
        "--scene-min-length-frames",
        type=int,
        default=DEFAULT_SCENE_MIN_LENGTH_FRAMES,
        help="Minimum PySceneDetect scene length in frames; default is 15.",
    )
    parser.add_argument(
        "--scene-snap-radius",
        type=float,
        default=DEFAULT_SCENE_SNAP_RADIUS_SECONDS,
        help="Maximum distance from a coarse boundary to a scene cut; default is 4s.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Lineup CSV. Defaults to predictions/lineup/lineup_segments.csv "
            "inside the transcript run directory."
        ),
    )
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--scene-slideshow",
        action="store_true",
        help="Use consecutive-slide scene pairing for broadcasts with roster slides.",
    )
    return parser


def _load_metadata(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise LineupDetectionError(f"Transcription metadata does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LineupDetectionError("Transcription metadata must be a JSON object.")
    return payload


def _request_summary(raw_response: dict[str, object]) -> dict[str, object]:
    raw_attempts = raw_response.get("_validation_attempts", [])
    attempts = raw_attempts if isinstance(raw_attempts, list) else []
    usage_totals: dict[str, int] = {}
    elapsed_seconds = 0.0
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        elapsed = attempt.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)):
            elapsed_seconds += float(elapsed)
        usage = attempt.get("usage")
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, int) and not isinstance(value, bool):
                usage_totals[str(key)] = usage_totals.get(str(key), 0) + value
    return {
        "request_count": len(attempts),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "usage": usage_totals,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(PROJECT_ROOT / ".env")
    try:
        if args.scene_threshold <= 0 or args.scene_min_length_frames <= 0:
            raise LineupDetectionError("Scene detector values must be positive.")
        if args.scene_snap_radius < 0:
            raise LineupDetectionError("--scene-snap-radius cannot be negative.")
        if args.max_coarse_duration <= 0:
            raise LineupDetectionError(
                "--max-coarse-duration must be positive."
            )

        transcript_path = resolve_project_path(args.transcript)
        transcription_metadata_path = resolve_project_path(
            args.transcription_metadata
            if args.transcription_metadata is not None
            else transcript_path.parent / "metadata.json"
        )
        transcription_metadata = _load_metadata(transcription_metadata_path)
        metadata_source = str(transcription_metadata.get("source_video", "")).strip()
        video_name = args.video or (
            Path(metadata_source).name if metadata_source else ""
        )
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
                    + float(transcription_metadata["requested_duration_seconds"]),
                ),
            )
        )

        output_path = resolve_project_path(
            args.output
            if args.output is not None
            else default_lineup_prediction_dir(
                transcript_path,
                source_video=metadata_source or None,
            )
            / "lineup_segments.csv"
        )
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

        api_key = os.getenv("QWEN_LINEUP_API_KEY") or os.getenv(
            "DASHSCOPE_API_KEY"
        )
        model = (
            args.model
            or os.getenv("QWEN_LINEUP_MODEL")
            or os.getenv("QWEN_TEXT_MODEL")
            or DEFAULT_TEXT_MODEL
        )
        base_url = (
            args.base_url
            or os.getenv("QWEN_LINEUP_BASE_URL")
            or DEFAULT_TEXT_BASE_URL
        )
        transcript = read_jsonl(transcript_path)
        print(
            f"Detecting lineup in {len(transcript)} coarse transcript chunk(s), "
            f"window {window_start:.3f}-{window_end:.3f}s with {model}"
        )
        result = QwenLineupDetector(
            api_key=api_key,
            expected_segment_count=args.expected_lineups,
            model=model,
            base_url=base_url,
            max_coarse_duration_seconds=args.max_coarse_duration,
        ).detect(
            transcript,
            video_name=video_name,
            window_start_seconds=window_start,
            window_end_seconds=window_end,
        )
        scene_refinement_ran = False
        source_video_path: Path | None = None
        if args.scene_snap and result.segments:
            source_value = args.source_video or (
                Path(metadata_source) if metadata_source else None
            )
            if source_value is None:
                raise LineupDetectionError(
                    "Source video is missing; pass --source-video or provide "
                    "it in metadata."
                )
            source_video_path = resolve_project_path(source_value)
            if not source_video_path.is_file():
                raise LineupDetectionError(
                    f"Source video does not exist: {source_video_path}"
                )
            margin = max(
                args.scene_snap_radius + 1.0,
                DEFAULT_MAX_GRAPHIC_DURATION_SECONDS + 1.0,
            )
            scan_start = max(
                0.0,
                min(segment.start_seconds for segment in result.segments) - margin,
            )
            scan_end = max(
                max(
                    segment.end_seconds + DEFAULT_MAX_AUDIO_END_LEAD_SECONDS,
                    segment.start_seconds
                    + DEFAULT_MAX_GRAPHIC_DURATION_SECONDS
                    + 1.0,
                )
                for segment in result.segments
            )
            source_duration = transcription_metadata.get("source_duration_seconds")
            if source_duration is not None:
                scan_end = min(scan_end, float(source_duration))
            if scan_end > scan_start:
                print(
                    f"Aligning coarse Qwen boundaries to PySceneDetect cuts within "
                    f"{args.scene_snap_radius:g}s"
                )
                cuts = detect_scene_cuts(
                    source_video_path,
                    start_seconds=scan_start,
                    end_seconds=scan_end,
                    threshold=args.scene_threshold,
                    min_scene_len_frames=args.scene_min_length_frames,
                )
                result = snap_lineup_result(
                    result,
                    cuts,
                    radius_seconds=args.scene_snap_radius,
                    slideshow_mode=args.scene_slideshow,
                )
                scene_refinement_ran = True
                scene_metadata = result.raw_response.get("scene_refinement")
                if isinstance(scene_metadata, dict):
                    scene_metadata["threshold"] = args.scene_threshold
                    scene_metadata["min_scene_length_frames"] = (
                        args.scene_min_length_frames
                    )
        request_summary = _request_summary(result.raw_response)
        write_lineup_csv(result, video_name=video_name, output_path=output_path)
        write_json(result.raw_response, raw_output_path)
        write_json(
            {
                "provider": "qwen",
                "lineup_api": "openai_compatible",
                "lineup_base_url": base_url,
                "model": result.model,
                "method": "coarse_transcript_qwen_text_plus_pyscenedetect",
                "uses_text_llm": True,
                "uses_visual_llm": False,
                "uses_pyscenedetect": scene_refinement_ran,
                "uses_second_pass_asr": False,
                "video": video_name,
                "source_video": str(source_video_path or metadata_source),
                "source_transcript": str(transcript_path),
                "transcription_metadata": str(transcription_metadata_path),
                "window_start_seconds": window_start,
                "window_end_seconds": window_end,
                "transcript_segment_count": len(transcript),
                "lineup_request_count": request_summary["request_count"],
                "lineup_request_elapsed_seconds": request_summary[
                    "elapsed_seconds"
                ],
                "lineup_token_usage": request_summary["usage"],
                "second_pass_request_count": 0,
                "lineup_segment_count": len(result.segments),
                "max_coarse_lineup_duration_seconds": (
                    args.max_coarse_duration
                ),
                "expected_lineup_segment_count": args.expected_lineups,
                "detection_status": result.status,
                "requires_review": (
                    args.expected_lineups is not None
                    and len(result.segments) != args.expected_lineups
                ),
                "timestamp_precision": (
                    "coarse_text_anchor_snapped_to_graphic_scene"
                    if scene_refinement_ran
                    else "coarse_text_anchor_within_sixty_second_chunk"
                ),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            detection_metadata_path,
        )
        print(f"Detected lineup segments: {len(result.segments)}")
        print(f"Detection status: {result.status}")
        print(
            "Qwen requests: "
            f"{request_summary['request_count']}; "
            f"elapsed: {request_summary['elapsed_seconds']}s; "
            f"usage: {request_summary['usage']}"
        )
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
    except QwenLineupValidationError as exc:
        failure_raw = exc.raw_response()
        try:
            write_json(failure_raw, raw_output_path)
            failure_summary = _request_summary(failure_raw)
            write_json(
                {
                    "provider": "qwen",
                    "model": model,
                    "detection_status": "failed_validation",
                    "requires_review": True,
                    "error": str(exc),
                    "lineup_request_count": failure_summary["request_count"],
                    "lineup_request_elapsed_seconds": failure_summary[
                        "elapsed_seconds"
                    ],
                    "lineup_token_usage": failure_summary["usage"],
                    "source_transcript": str(transcript_path),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                detection_metadata_path,
            )
        except (OSError, TypeError, ValueError):
            pass
        print(f"Lineup detection failed: {exc}", file=sys.stderr)
        return 1
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
