"""CLI: detect coarse lineups, confirm them with OCR, then export timestamps."""

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
    DEFAULT_SCENE_MIN_LENGTH_SECONDS,
    DEFAULT_SCENE_SNAP_RADIUS_SECONDS,
    DEFAULT_SCENE_THRESHOLD,
)
from lineup.schema import parse_detection_result
from lineup.utils import (
    PROJECT_ROOT,
    default_lineup_prediction_dir,
    resolve_project_path,
)
from lineup.visual_refinement import (
    DEFAULT_OCR_PYTHON,
    MAX_SCAN_SECONDS,
    refine_with_visual_ocr,
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
            "Qwen call, tiny OCR over scene representatives, and PySceneDetect."
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
        "--no-ocr",
        action="store_false",
        dest="visual_ocr",
        help="Keep Qwen coarse boundaries without OCR/PySceneDetect refinement.",
    )
    parser.add_argument(
        "--reuse-coarse",
        action="store_true",
        help=(
            "Reuse lineup_segments from the existing raw output and rerun only "
            "OCR/PySceneDetect. Requires --overwrite and avoids another Qwen call."
        ),
    )
    parser.add_argument(
        "--ocr-python",
        type=Path,
        default=Path(os.getenv("LINEUP_OCR_PYTHON", str(DEFAULT_OCR_PYTHON))),
        help="Python executable containing PaddleOCR; default is .venv-ocr.",
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
        default=None,
        help=(
            "Explicit PySceneDetect minimum scene length in frames. By default "
            "the FPS-independent --scene-min-length-seconds value is used."
        ),
    )
    parser.add_argument(
        "--scene-min-length-seconds",
        type=float,
        default=DEFAULT_SCENE_MIN_LENGTH_SECONDS,
        help="FPS-independent minimum scene length; default is 0.5 seconds.",
    )
    parser.add_argument(
        "--scene-snap-radius",
        type=float,
        default=DEFAULT_SCENE_SNAP_RADIUS_SECONDS,
        help="Maximum final OCR-boundary adjustment to a scene cut; default is 4s.",
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
        if args.scene_threshold <= 0 or args.scene_min_length_seconds <= 0:
            raise LineupDetectionError("Scene detector values must be positive.")
        if (
            args.scene_min_length_frames is not None
            and args.scene_min_length_frames <= 0
        ):
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
        metadata_window_end = float(
            transcription_metadata.get(
                "processed_end_seconds",
                transcription_metadata.get(
                    "requested_end_seconds",
                    window_start
                    + float(transcription_metadata["requested_duration_seconds"]),
                ),
            )
        )
        window_end = min(metadata_window_end, MAX_SCAN_SECONDS)
        if window_start >= window_end:
            raise LineupDetectionError(
                "The transcript window does not overlap the first 600 seconds."
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
        transcript = tuple(
            segment
            for segment in read_jsonl(transcript_path)
            if segment.start_seconds < window_end
            and segment.end_seconds <= window_end + 1e-6
        )
        if args.reuse_coarse:
            if not args.overwrite:
                raise LineupDetectionError(
                    "--reuse-coarse requires --overwrite because it reads and "
                    "replaces the existing detection outputs."
                )
            if not raw_output_path.is_file():
                raise LineupDetectionError(
                    f"Existing coarse raw output does not exist: {raw_output_path}"
                )
            saved_payload = json.loads(raw_output_path.read_text(encoding="utf-8"))
            if not isinstance(saved_payload, dict):
                raise LineupDetectionError(
                    "Existing coarse raw output must be a JSON object."
                )
            coarse_payload = {
                key: value
                for key, value in saved_payload.items()
                if key not in {"visual_refinement", "_review_reasons"}
            }
            evidence_bounds = {
                segment.segment_id: (segment.start_seconds, segment.end_seconds)
                for segment in transcript
            }
            evidence_texts = {
                segment.segment_id: segment.text for segment in transcript
            }
            result = parse_detection_result(
                coarse_payload,
                model=model,
                known_evidence=evidence_bounds,
                evidence_bounds=evidence_bounds,
                evidence_texts=evidence_texts,
                window_start=window_start,
                window_end=window_end,
                max_segment_duration_seconds=args.max_coarse_duration,
                # The saved response already passed validation. Older runs
                # predate the explicit kickoff field, so do not reject them
                # when rerunning only the visual stage.
                require_kickoff_field=False,
            )
            print(
                f"Reusing {len(result.segments)} coarse lineup candidate(s); "
                "no Qwen request"
            )
        else:
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
                require_kickoff_field=True,
            ).detect(
                transcript,
                video_name=video_name,
                window_start_seconds=window_start,
                window_end_seconds=window_end,
            )
        visual_refinement_ran = False
        source_video_path: Path | None = None
        if args.visual_ocr:
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
            source_duration = transcription_metadata.get("source_duration_seconds")
            visual_scan_end = min(
                MAX_SCAN_SECONDS,
                float(source_duration) if source_duration is not None else window_end,
            )
            print(
                "Confirming lineup graphics with scene-level tiny OCR "
                f"inside the first {visual_scan_end:g}s"
            )
            result = refine_with_visual_ocr(
                result,
                video_path=source_video_path,
                scan_end_seconds=visual_scan_end,
                expected_count=args.expected_lineups,
                scene_threshold=args.scene_threshold,
                scene_min_length_frames=args.scene_min_length_frames,
                scene_min_length_seconds=args.scene_min_length_seconds,
                scene_snap_radius_seconds=args.scene_snap_radius,
                ocr_python=args.ocr_python,
            )
            visual_refinement_ran = True
        request_summary = _request_summary(result.raw_response)
        visual_diagnostics = result.raw_response.get("visual_refinement", {})
        if not isinstance(visual_diagnostics, dict):
            visual_diagnostics = {}
        review_reasons = result.raw_response.get("_review_reasons", [])
        if not isinstance(review_reasons, list):
            review_reasons = []
        requires_review = bool(review_reasons) or (
            args.expected_lineups is not None
            and len(result.segments) != args.expected_lineups
        )
        write_lineup_csv(result, video_name=video_name, output_path=output_path)
        write_json(result.raw_response, raw_output_path)
        write_json(
            {
                "provider": "qwen",
                "lineup_api": "openai_compatible",
                "lineup_base_url": base_url,
                "model": result.model,
                "method": (
                    "coarse_transcript_qwen_plus_scene_tiny_ocr_pyscenedetect"
                    if visual_refinement_ran
                    else "coarse_transcript_qwen_text"
                ),
                "uses_text_llm": True,
                "coarse_qwen_response_reused": args.reuse_coarse,
                "uses_visual_llm": False,
                "uses_ocr": visual_refinement_ran,
                "ocr_detection_model": (
                    "PP-OCRv6_tiny_det" if visual_refinement_ran else None
                ),
                "ocr_recognition_model": (
                    "PP-OCRv6_tiny_rec" if visual_refinement_ran else None
                ),
                "ocr_global_fallback_used": bool(
                    visual_diagnostics.get("fallback_used")
                ),
                "uses_pyscenedetect": visual_refinement_ran,
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
                    requires_review
                ),
                "review_reasons": review_reasons,
                "timestamp_precision": (
                    "scene_representative_ocr_snapped_to_graphic_scene"
                    if visual_refinement_ran
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
