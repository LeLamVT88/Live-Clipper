"""Run video extraction, MobileNet detection, and lineup OCR in order."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = PROJECT_ROOT / "data" / "data_test"
MOBILENET_DIR = PROJECT_ROOT / "outputs" / "predictions" / "mobilenet"
MODEL_PATH = MOBILENET_DIR / "mobilenet_v3_small_lineup.pt"
OCR_PYTHON = PROJECT_ROOT / ".venv-ocr" / "bin" / "python"
RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


@dataclass(frozen=True)
class RunPaths:
    root: Path
    clips: Path
    mobilenet_frames: Path
    ocr_frames: Path
    selected_frames: Path
    mobilenet: Path
    ocr: Path

    @classmethod
    def for_video(cls, video: str) -> RunPaths:
        root = RUNS_DIR / Path(video).stem
        return cls(
            root=root,
            clips=root / "clips",
            mobilenet_frames=root / "frames" / "mobilenet",
            ocr_frames=root / "frames" / "ocr",
            selected_frames=root / "frames" / "ocr_selected",
            mobilenet=root / "predictions" / "mobilenet",
            ocr=root / "predictions" / "ocr",
        )

    @property
    def metadata_csv(self) -> Path:
        return self.mobilenet / "extracted_frames.csv"

    @property
    def inference_csv(self) -> Path:
        return self.mobilenet / "mobilenet_v3_small_inference.csv"

    @property
    def segments_csv(self) -> Path:
        return self.mobilenet / "lineup_segments.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete MobileNet-to-OCR lineup pipeline."
    )
    parser.add_argument(
        "--video",
        action="append",
        default=[],
        help="Video name in data/data_test. Repeat for multiple videos; omit for all.",
    )
    parser.add_argument("--start", default="0", help="Extraction start timestamp.")
    time_range = parser.add_mutually_exclusive_group()
    time_range.add_argument(
        "--duration",
        help="Extraction duration. The extractor defaults to 00:10:00.",
    )
    time_range.add_argument("--end", help="Extraction end timestamp.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--ocr-python",
        type=Path,
        default=OCR_PYTHON,
        help="Python environment containing PaddleOCR.",
    )
    parser.add_argument("--merge-gap-seconds", type=float, default=6.0)
    parser.add_argument("--min-duration-seconds", type=float, default=8.0)
    return parser


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def available_videos() -> list[str]:
    return sorted(
        path.relative_to(VIDEO_DIR).as_posix()
        for path in VIDEO_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def selected_videos(args: argparse.Namespace) -> list[str]:
    available = available_videos()
    if not args.video:
        return available

    selected: list[str] = []
    for name in args.video:
        requested = Path(name).as_posix()
        matches = [
            video
            for video in available
            if video == requested or Path(video).name == Path(requested).name
        ]
        if not matches:
            raise ValueError(f"Video file not found: {name}")
        if len(matches) > 1:
            raise ValueError(
                f"Video name is ambiguous: {name}. Use one of: "
                + ", ".join(matches)
            )
        if matches[0] not in selected:
            selected.append(matches[0])
    return selected


def validate_inputs(args: argparse.Namespace) -> str | None:
    if not project_path(args.model).is_file():
        return f"MobileNet checkpoint does not exist: {project_path(args.model)}"
    if not project_path(args.ocr_python).is_file():
        return f"OCR Python does not exist: {project_path(args.ocr_python)}"
    if not VIDEO_DIR.is_dir():
        return f"Video directory does not exist: {VIDEO_DIR}"
    try:
        videos = selected_videos(args)
    except ValueError as exc:
        return str(exc)
    if not videos:
        return f"No video files found in: {VIDEO_DIR}"
    return None


def run_stage(position: int, name: str, command: list[str]) -> int:
    print(f"\n[{position}/5] {name}", flush=True)
    try:
        return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode
    except OSError as exc:
        print(f"Cannot start {name}: {exc}", file=sys.stderr)
        return 1


def segments_exist(path: Path) -> bool:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return next(csv.DictReader(file), None) is not None


def command(
    script: str,
    *arguments: object,
    interpreter: str | Path | None = None,
) -> list[str]:
    return [
        str(interpreter or sys.executable),
        str(PROJECT_ROOT / "src" / script),
        *(str(argument) for argument in arguments),
    ]


def build_commands(
    args: argparse.Namespace,
    video: str,
    paths: RunPaths,
) -> list[tuple[str, list[str]]]:
    extract = command(
        "lineup/extract_frames.py",
        "--input-dir",
        VIDEO_DIR,
        "--output-dir",
        paths.mobilenet_frames,
        "--metadata-csv",
        paths.metadata_csv,
        "--skip-video-metadata",
        "--video",
        video,
        "--start",
        args.start,
    )
    if args.duration is not None:
        extract.extend(("--duration", args.duration))
    if args.end is not None:
        extract.extend(("--end", args.end))

    return [
        ("Extracting MobileNet frames", extract),
        (
            "Running MobileNet inference",
            command(
                "lineup/predict_mobilenet.py",
                "--input-csv",
                paths.metadata_csv,
                "--model",
                project_path(args.model),
                "--output-csv",
                paths.inference_csv,
                "--device",
                args.device,
            ),
        ),
        (
            "Aggregating lineup segments",
            command(
                "lineup/aggregate.py",
                "--predictions-csv",
                paths.inference_csv,
                "--output-csv",
                paths.segments_csv,
                "--merge-gap-seconds",
                args.merge_gap_seconds,
                "--min-duration-seconds",
                args.min_duration_seconds,
                "--expected-segments-per-video",
                2,
            ),
        ),
        (
            "Exporting lineup clips",
            command(
                "lineup/export_clips.py",
                "--segments-csv",
                paths.segments_csv,
                "--video-dir",
                VIDEO_DIR,
                "--output-dir",
                paths.clips,
                "--overwrite",
            ),
        ),
        (
            "Running OCR and lineup resolver",
            command(
                "ocr/run_pipeline.py",
                "--segments-csv",
                paths.segments_csv,
                "--video-dir",
                VIDEO_DIR,
                "--frames-dir",
                paths.ocr_frames,
                "--selected-frames-dir",
                paths.selected_frames,
                "--frames-csv",
                paths.ocr / "ocr_frames.csv",
                "--scout-output-csv",
                paths.ocr / "ocr_scout_detections.csv",
                "--selected-frames-csv",
                paths.ocr / "ocr_selected_frames.csv",
                "--selection-diagnostics-csv",
                paths.ocr / "ocr_frame_selection_diagnostics.csv",
                "--output-csv",
                paths.ocr / "ocr_raw_detections.csv",
                "--resolved-output-csv",
                paths.ocr / "resolved_lineups.csv",
                "--resolved-diagnostics-csv",
                paths.ocr / "resolved_lineups_diagnostics.csv",
                "--attempts-csv",
                paths.ocr / "pipeline_attempts.csv",
                interpreter=project_path(args.ocr_python),
            ),
        ),
    ]


def run_video(args: argparse.Namespace, video: str) -> int:
    paths = RunPaths.for_video(video)
    print(f"\nVideo: {video}\nOutput: {paths.root}")
    stages = build_commands(args, video, paths)
    for position, (name, command) in enumerate(stages[:3], start=1):
        if return_code := run_stage(position, name, command):
            print(f"Pipeline stopped at stage {position}: {name}", file=sys.stderr)
            return return_code

    try:
        has_segments = segments_exist(paths.segments_csv)
    except (OSError, csv.Error) as exc:
        print(f"Cannot read generated segments: {exc}", file=sys.stderr)
        return 1
    if not has_segments:
        print("\nNo lineup segments detected; OCR was skipped.")
        return 0

    clip_code = run_stage(4, *stages[3])
    if clip_code:
        print("Pipeline stopped during clip export.", file=sys.stderr)
        return clip_code

    ocr_code = run_stage(5, *stages[4])
    if ocr_code == 2:
        print("OCR completed, but some segments did not pass the quality gate.")
    elif ocr_code:
        print("Pipeline stopped during OCR.", file=sys.stderr)
    else:
        print(f"All outputs saved to: {paths.root}")
    return ocr_code


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    error = validate_inputs(args)
    if error:
        print(f"Pipeline cannot start: {error}", file=sys.stderr)
        return 1

    overall_code = 0
    videos = selected_videos(args)
    for position, video in enumerate(videos, start=1):
        print(f"\n=== Video {position}/{len(videos)} ===")
        return_code = run_video(args, video)
        if return_code not in (0, 2):
            return return_code
        overall_code = max(overall_code, return_code)
    return overall_code


if __name__ == "__main__":
    raise SystemExit(main())
