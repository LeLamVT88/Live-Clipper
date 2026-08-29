"""CLI: extract video audio and transcribe it with Qwen ASR."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from lineup.utils import (
    PROJECT_ROOT,
    default_run_dir,
    resolve_project_path,
    timestamp_to_seconds,
)
from transcript.audio import (
    AudioChunk,
    AudioExtractionError,
    MediaInfo,
    extract_audio,
    plan_audio_chunks,
    probe_media,
    wav_duration_seconds,
)
from transcript.qwen import (
    DEFAULT_ASR_MODEL,
    DEFAULT_BASE_URL,
    QwenTranscriber,
    QwenTranscriptionError,
)
from transcript.schema import (
    TranscriptSegment,
    TranscriptValidationError,
    TranscriptionResult,
    read_jsonl,
    write_json,
    write_jsonl,
)


DEFAULT_DURATION_SECONDS = 600.0
DEFAULT_CHUNK_DURATION_SECONDS = 60.0
DEFAULT_MAX_CHUNKS = 10
DEFAULT_WORKERS = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract video audio and transcribe it with Qwen ASR."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start", default="0", help="Video start timestamp.")
    parser.add_argument(
        "--duration",
        default=str(int(DEFAULT_DURATION_SECONDS)),
        help="Total video duration to transcribe; default is 600 seconds.",
    )
    parser.add_argument(
        "--chunk-duration",
        default=str(int(DEFAULT_CHUNK_DURATION_SECONDS)),
        help=(
            "Maximum duration of each Qwen request; default is 60 seconds "
            "(10 chunks for the default 600-second window)."
        ),
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=DEFAULT_MAX_CHUNKS,
        help="Maximum number of audio chunks; default is 10.",
    )
    parser.add_argument(
        "--language",
        default="auto",
        help="Expected commentary language, for example vi, en, or auto.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Qwen ASR model; defaults to QWEN_ASR_MODEL or qwen3-asr-flash.",
    )
    parser.add_argument(
        "--base-url",
        help="DashScope API base URL; defaults to the Singapore endpoint.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Concurrent Qwen ASR requests; default is 4.",
    )
    parser.add_argument(
        "--context",
        help="Optional short context such as team and player names for Qwen ASR.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Run directory. By default, mirrors the video path below "
            "data/raw_data directly under outputs."
        ),
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse completed per-chunk transcripts and existing audio after "
            "an interrupted run."
        ),
    )
    return parser


def fit_processing_duration(
    *,
    media: MediaInfo,
    start_seconds: float,
    requested_duration_seconds: float,
    chunk_duration_seconds: float,
    max_chunks: int,
) -> float:
    """Clamp one scan window to the source and the configured chunk budget."""
    media.validate()
    values = (
        start_seconds,
        requested_duration_seconds,
        chunk_duration_seconds,
    )
    if not all(math.isfinite(value) for value in values):
        raise AudioExtractionError("Processing window values must be finite.")
    if start_seconds < 0:
        raise AudioExtractionError("Processing start must be non-negative.")
    if requested_duration_seconds <= 0 or chunk_duration_seconds <= 0:
        raise AudioExtractionError(
            "Requested duration and chunk duration must be positive."
        )
    if not media.has_video:
        raise AudioExtractionError("Input has no video stream.")
    if not media.has_audio:
        raise AudioExtractionError("Input has no audio stream to transcribe.")
    if max_chunks <= 0:
        raise AudioExtractionError("max_chunks must be positive.")
    if start_seconds >= media.duration_seconds:
        raise AudioExtractionError(
            "Processing start is at or beyond the end of the input video."
        )
    available = media.duration_seconds - start_seconds
    chunk_budget = chunk_duration_seconds * max_chunks
    effective = min(requested_duration_seconds, available, chunk_budget)
    if not math.isfinite(effective) or effective <= 0:
        raise AudioExtractionError("Effective processing duration must be positive.")
    return effective


def load_cached_chunk(
    *,
    transcript_path: Path,
    raw_response_path: Path,
    source_chunk: str,
    chunk_start_seconds: float,
) -> tuple[tuple[TranscriptSegment, ...], dict[str, object]]:
    segments = read_jsonl(transcript_path)
    if any(
        segment.source_chunk != source_chunk
        or abs(segment.chunk_start_seconds - chunk_start_seconds) > 1e-6
        for segment in segments
    ):
        raise TranscriptValidationError(
            f"Cached chunk does not match the current plan: {source_chunk}"
        )
    payload = json.loads(raw_response_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TranscriptValidationError(
            f"Cached raw response must be an object: {raw_response_path}"
        )
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list) or len(raw_segments) != len(segments):
        raise TranscriptValidationError(
            f"Cached chunk is incomplete: {source_chunk}"
        )
    return segments, payload


def _chunk_paths(run_dir: Path, chunk_id: str) -> tuple[Path, Path, Path]:
    transcript_dir = run_dir / "predictions" / "transcript"
    return (
        run_dir / "audio" / f"{chunk_id}.wav",
        transcript_dir / "raw" / f"{chunk_id}.json",
        transcript_dir / "chunks" / f"{chunk_id}.jsonl",
    )


def _process_chunk(
    args: argparse.Namespace,
    *,
    chunk: AudioChunk,
    position: int,
    total: int,
    video_path: Path,
    run_dir: Path,
    transcriber: QwenTranscriber,
) -> tuple[TranscriptionResult, dict[str, object]]:
    processing_started = time.monotonic()
    audio_path, raw_path, transcript_path = _chunk_paths(
        run_dir, chunk.chunk_id
    )
    complete = args.resume and all(
        path.is_file() for path in (audio_path, raw_path, transcript_path)
    )
    if complete:
        print(f"[{position}/{total}] Reusing completed {chunk.chunk_id}")
        segments, payload = load_cached_chunk(
            transcript_path=transcript_path,
            raw_response_path=raw_path,
            source_chunk=chunk.chunk_id,
            chunk_start_seconds=chunk.start_seconds,
        )
        result = TranscriptionResult(transcriber.model, segments, payload)
        duration = wav_duration_seconds(audio_path)
    else:
        if args.resume and audio_path.is_file():
            print(f"[{position}/{total}] Reusing audio for {chunk.chunk_id}")
        else:
            print(
                f"[{position}/{total}] Extracting {chunk.chunk_id}: "
                f"{chunk.start_seconds:.3f}-{chunk.end_seconds:.3f}s"
            )
            extract_audio(
                video_path,
                audio_path,
                start_seconds=chunk.start_seconds,
                duration_seconds=chunk.duration_seconds,
                ffmpeg=args.ffmpeg,
                overwrite=args.overwrite,
            )
        duration = wav_duration_seconds(audio_path)
        print(
            f"[{position}/{total}] Transcribing {chunk.chunk_id} "
            f"with {transcriber.model}"
        )
        result = transcriber.transcribe(
            audio_path,
            audio_duration_seconds=duration,
            language=args.language,
            source_chunk=chunk.chunk_id,
            chunk_start_seconds=chunk.start_seconds,
            context=args.context,
            raw_response_callback=lambda payload: write_json(payload, raw_path),
        )
        write_jsonl(result.segments, transcript_path)

    result.validate()
    metadata = {
        "source_chunk": chunk.chunk_id,
        "start_seconds": chunk.start_seconds,
        "requested_duration_seconds": chunk.duration_seconds,
        "audio_duration_seconds": round(duration, 3),
        "segment_count": len(result.segments),
        "audio_path": str(audio_path),
        "raw_response_path": str(raw_path),
        "chunk_transcript_path": str(transcript_path),
        "processing_seconds": round(time.monotonic() - processing_started, 3),
    }
    return result, metadata


def _transcribe_chunks(
    args: argparse.Namespace,
    *,
    api_key: str,
    model: str,
    base_url: str,
    video_path: Path,
    run_dir: Path,
    start_seconds: float,
    duration_seconds: float,
    chunk_duration_seconds: float,
) -> tuple[TranscriptionResult, list[dict[str, object]], int]:
    chunks = plan_audio_chunks(
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        chunk_duration_seconds=chunk_duration_seconds,
    )
    transcript_dir = run_dir / "predictions" / "transcript"
    expected = [
        transcript_dir / "transcript.jsonl",
        transcript_dir / "qwen_raw_response.json",
        transcript_dir / "metadata.json",
    ]
    expected.extend(
        path
        for chunk in chunks
        for path in _chunk_paths(run_dir, chunk.chunk_id)
    )
    existing = next((path for path in expected if path.exists()), None)
    if existing and not args.overwrite and not args.resume:
        raise FileExistsError(f"Output already exists; pass --overwrite: {existing}")

    legacy_audio = run_dir / "audio" / "transcription_input.wav"
    if args.overwrite and legacy_audio.exists():
        legacy_audio.unlink()
    print(
        f"Processing {video_path.name}: {start_seconds:.3f}-"
        f"{start_seconds + duration_seconds:.3f}s in {len(chunks)} chunk(s)"
    )

    transcriber = QwenTranscriber(
        api_key=api_key,
        model=model,
        base_url=base_url,
    )
    completed: list[
        tuple[AudioChunk, TranscriptionResult, dict[str, object]]
    ] = []
    worker_count = min(args.workers, len(chunks))
    print(f"Using {worker_count} concurrent Qwen ASR worker(s)")
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _process_chunk,
                args,
                chunk=chunk,
                position=position,
                total=len(chunks),
                video_path=video_path,
                run_dir=run_dir,
                transcriber=transcriber,
            ): chunk
            for position, chunk in enumerate(chunks, start=1)
        }
        for future in as_completed(futures):
            chunk = futures[future]
            result, chunk_metadata = future.result()
            completed.append((chunk, result, chunk_metadata))

    completed.sort(key=lambda item: item[0].start_seconds)
    segments: list[TranscriptSegment] = []
    raw_chunks: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []
    for chunk, result, chunk_metadata in completed:
        segments.extend(result.segments)
        raw_chunks.append(
            {
                "source_chunk": chunk.chunk_id,
                "chunk_start_seconds": chunk.start_seconds,
                "response": result.raw_response,
            }
        )
        metadata.append(chunk_metadata)

    segments.sort(key=lambda segment: (segment.start_seconds, segment.end_seconds))
    combined = TranscriptionResult(
        model=model,
        segments=tuple(segments),
        raw_response={"model": model, "chunks": raw_chunks},
    )
    combined.validate()
    return combined, metadata, len(chunks)


def _write_run_outputs(
    result: TranscriptionResult,
    *,
    args: argparse.Namespace,
    video_path: Path,
    run_dir: Path,
    start_seconds: float,
    requested_duration_seconds: float,
    processed_duration_seconds: float,
    chunk_duration_seconds: float,
    max_chunks: int,
    media: MediaInfo,
    chunk_count: int,
    chunk_metadata: list[dict[str, object]],
) -> tuple[Path, Path]:
    transcript_dir = run_dir / "predictions" / "transcript"
    transcript_path = transcript_dir / "transcript.jsonl"
    metadata_path = transcript_dir / "metadata.json"
    write_jsonl(result.segments, transcript_path)
    write_json(result.raw_response, transcript_dir / "qwen_raw_response.json")
    write_json(
        {
            "provider": "qwen",
            "model": result.model,
            "source_video": str(video_path),
            "source_start_seconds": start_seconds,
            "source_duration_seconds": round(media.duration_seconds, 3),
            "source_audio_stream_count": media.audio_stream_count,
            "source_video_stream_count": media.video_stream_count,
            "requested_duration_seconds": requested_duration_seconds,
            "requested_end_seconds": start_seconds + requested_duration_seconds,
            "processed_duration_seconds": processed_duration_seconds,
            "processed_end_seconds": start_seconds + processed_duration_seconds,
            "chunk_duration_seconds": chunk_duration_seconds,
            "max_chunks": max_chunks,
            "chunk_count": chunk_count,
            "workers": args.workers,
            "total_audio_duration_seconds": round(
                sum(float(item["audio_duration_seconds"]) for item in chunk_metadata),
                3,
            ),
            "timestamp_basis": "absolute_video_seconds_from_chunk_bounds",
            "timestamp_precision": "chunk",
            "coarse_boundary_alignment": (
                "normalized_character_position_within_chunk"
            ),
            "language": args.language,
            "segment_count": len(result.segments),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "transcript_path": str(transcript_path),
            "chunks": chunk_metadata,
        },
        metadata_path,
    )
    return transcript_path, metadata_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(PROJECT_ROOT / ".env")
    try:
        start_seconds = timestamp_to_seconds(args.start)
        duration_seconds = timestamp_to_seconds(args.duration)
        chunk_duration_seconds = timestamp_to_seconds(args.chunk_duration)
        if duration_seconds <= 0:
            raise ValueError("--duration must be positive.")
        if chunk_duration_seconds <= 0:
            raise ValueError("--chunk-duration must be positive.")
        if args.workers <= 0:
            raise ValueError("--workers must be positive.")
        if args.max_chunks <= 0:
            raise ValueError("--max-chunks must be positive.")
        if args.resume and args.overwrite:
            raise ValueError("--resume and --overwrite cannot be used together.")

        video_path = resolve_project_path(args.video)
        media = probe_media(video_path, ffprobe=args.ffprobe)
        processed_duration_seconds = fit_processing_duration(
            media=media,
            start_seconds=start_seconds,
            requested_duration_seconds=duration_seconds,
            chunk_duration_seconds=chunk_duration_seconds,
            max_chunks=args.max_chunks,
        )
        if processed_duration_seconds < duration_seconds - 1e-6:
            print(
                "Processing window clamped from "
                f"{duration_seconds:.3f}s to {processed_duration_seconds:.3f}s "
                "by source duration or chunk budget."
            )

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise QwenTranscriptionError(
                "Missing API key. Set DASHSCOPE_API_KEY in the project .env file."
            )
        model = args.model or os.getenv("QWEN_ASR_MODEL") or DEFAULT_ASR_MODEL
        base_url = (
            args.base_url or os.getenv("DASHSCOPE_BASE_URL") or DEFAULT_BASE_URL
        )
        run_dir = (
            resolve_project_path(args.output_dir)
            if args.output_dir is not None
            else default_run_dir(video_path)
        )
        result, chunk_metadata, chunk_count = _transcribe_chunks(
            args,
            api_key=api_key,
            model=model,
            base_url=base_url,
            video_path=video_path,
            run_dir=run_dir,
            start_seconds=start_seconds,
            duration_seconds=processed_duration_seconds,
            chunk_duration_seconds=chunk_duration_seconds,
        )
        transcript_path, metadata_path = _write_run_outputs(
            result,
            args=args,
            video_path=video_path,
            run_dir=run_dir,
            start_seconds=start_seconds,
            requested_duration_seconds=duration_seconds,
            processed_duration_seconds=processed_duration_seconds,
            chunk_duration_seconds=chunk_duration_seconds,
            max_chunks=args.max_chunks,
            media=media,
            chunk_count=chunk_count,
            chunk_metadata=chunk_metadata,
        )
        print(f"Transcript segments: {len(result.segments)}")
        print(f"Transcript: {transcript_path}")
        print(f"Metadata: {metadata_path}")
        return 0
    except (
        AudioExtractionError,
        FileExistsError,
        QwenTranscriptionError,
        json.JSONDecodeError,
        OSError,
        TranscriptValidationError,
        ValueError,
    ) as exc:
        print(f"Transcription failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
