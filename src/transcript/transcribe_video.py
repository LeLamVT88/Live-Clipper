"""CLI: extract video audio and create a timestamped Gemini transcript."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from lineup.utils import safe_stem, timestamp_to_seconds
from transcript.audio import (
    AudioExtractionError,
    extract_audio,
    plan_audio_chunks,
    wav_duration_seconds,
)
from transcript.gemini import DEFAULT_MODEL, GeminiTranscriber, GeminiTranscriptionError
from transcript.schema import (
    TranscriptSegment,
    TranscriptValidationError,
    TranscriptionResult,
    write_json,
    write_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DURATION_SECONDS = 600.0
DEFAULT_CHUNK_DURATION_SECONDS = 60.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract video audio and transcribe it with timestamped Gemini output."
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
        help="Maximum duration of each Gemini request; default is 60 seconds.",
    )
    parser.add_argument(
        "--language",
        default="auto",
        help="Expected commentary language, for example vi, en, or auto.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Gemini model; defaults to GEMINI_MODEL or gemini-3.6-flash.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Run directory; defaults to outputs/runs/<video-stem>.",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
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
        start_seconds = timestamp_to_seconds(args.start)
        duration_seconds = timestamp_to_seconds(args.duration)
        chunk_duration_seconds = timestamp_to_seconds(args.chunk_duration)
        if duration_seconds <= 0:
            raise ValueError("--duration must be positive.")
        if chunk_duration_seconds <= 0:
            raise ValueError("--chunk-duration must be positive.")

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise GeminiTranscriptionError(
                "Missing API key. Copy .env.example to .env and set GEMINI_API_KEY."
            )
        model = args.model or os.getenv("GEMINI_MODEL") or DEFAULT_MODEL
        video_path = resolve_project_path(args.video)
        run_dir = (
            resolve_project_path(args.output_dir)
            if args.output_dir is not None
            else PROJECT_ROOT / "outputs" / "runs" / safe_stem(video_path.stem)
        )
        audio_dir = run_dir / "audio"
        transcript_dir = run_dir / "predictions" / "transcript"
        raw_response_dir = transcript_dir / "raw"
        transcript_path = transcript_dir / "transcript.jsonl"
        raw_response_path = transcript_dir / "gemini_raw_response.json"
        metadata_path = transcript_dir / "metadata.json"

        chunks = plan_audio_chunks(
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            chunk_duration_seconds=chunk_duration_seconds,
        )
        chunk_audio_paths = {
            chunk.chunk_id: audio_dir / f"{chunk.chunk_id}.wav" for chunk in chunks
        }
        chunk_raw_paths = {
            chunk.chunk_id: raw_response_dir / f"{chunk.chunk_id}.json"
            for chunk in chunks
        }
        expected_outputs = [transcript_path, raw_response_path, metadata_path]
        expected_outputs.extend(chunk_audio_paths.values())
        expected_outputs.extend(chunk_raw_paths.values())
        for output in expected_outputs:
            if output.exists() and not args.overwrite:
                raise FileExistsError(
                    f"Output already exists; pass --overwrite: {output}"
                )

        legacy_audio_path = audio_dir / "transcription_input.wav"
        if args.overwrite and legacy_audio_path.exists():
            legacy_audio_path.unlink()

        print(
            f"Processing {video_path.name}: {start_seconds:.3f}-"
            f"{start_seconds + duration_seconds:.3f}s in {len(chunks)} chunk(s)"
        )
        transcriber = GeminiTranscriber(api_key=api_key, model=model)
        all_segments: list[TranscriptSegment] = []
        raw_chunks: list[dict[str, object]] = []
        chunk_metadata: list[dict[str, object]] = []

        for position, chunk in enumerate(chunks, start=1):
            audio_path = chunk_audio_paths[chunk.chunk_id]
            print(
                f"[{position}/{len(chunks)}] Extracting {chunk.chunk_id}: "
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
            actual_audio_duration = wav_duration_seconds(audio_path)

            print(
                f"[{position}/{len(chunks)}] Transcribing {chunk.chunk_id} "
                f"with {model}"
            )
            result = transcriber.transcribe(
                audio_path,
                audio_duration_seconds=actual_audio_duration,
                language=args.language,
                source_chunk=chunk.chunk_id,
                chunk_start_seconds=chunk.start_seconds,
                raw_response_callback=lambda payload, path=chunk_raw_paths[
                    chunk.chunk_id
                ]: write_json(payload, path),
            )
            all_segments.extend(result.segments)
            raw_chunks.append(
                {
                    "source_chunk": chunk.chunk_id,
                    "chunk_start_seconds": chunk.start_seconds,
                    "response": result.raw_response,
                }
            )
            chunk_metadata.append(
                {
                    "source_chunk": chunk.chunk_id,
                    "start_seconds": chunk.start_seconds,
                    "requested_duration_seconds": chunk.duration_seconds,
                    "audio_duration_seconds": round(actual_audio_duration, 3),
                    "segment_count": len(result.segments),
                    "audio_path": str(audio_path),
                    "raw_response_path": str(chunk_raw_paths[chunk.chunk_id]),
                }
            )

        all_segments.sort(key=lambda segment: (segment.start_seconds, segment.end_seconds))
        combined_result = TranscriptionResult(
            model=model,
            segments=tuple(all_segments),
            raw_response={"model": model, "chunks": raw_chunks},
        )
        combined_result.validate()
        write_jsonl(combined_result.segments, transcript_path)
        write_json(combined_result.raw_response, raw_response_path)
        write_json(
            {
                "provider": "gemini",
                "model": combined_result.model,
                "source_video": str(video_path),
                "source_start_seconds": start_seconds,
                "requested_duration_seconds": duration_seconds,
                "requested_end_seconds": start_seconds + duration_seconds,
                "chunk_duration_seconds": chunk_duration_seconds,
                "chunk_count": len(chunks),
                "total_audio_duration_seconds": round(
                    sum(float(item["audio_duration_seconds"]) for item in chunk_metadata),
                    3,
                ),
                "timestamp_basis": "absolute_video_seconds",
                "language": args.language,
                "segment_count": len(combined_result.segments),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "transcript_path": str(transcript_path),
                "chunks": chunk_metadata,
            },
            metadata_path,
        )
        print(f"Transcript segments: {len(combined_result.segments)}")
        print(f"Transcript: {transcript_path}")
        print(f"Metadata: {metadata_path}")
        return 0
    except (
        AudioExtractionError,
        FileExistsError,
        GeminiTranscriptionError,
        OSError,
        TranscriptValidationError,
        ValueError,
    ) as exc:
        print(f"Transcription failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
