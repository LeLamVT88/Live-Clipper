"""Configuration and CLI arguments for the lineup pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path

from .frames import PROJECT_ROOT, LineupOCRError, resolve_project_path


OCR_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions" / "ocr"
DEFAULTS = {
    "segments_csv": (
        PROJECT_ROOT
        / "outputs"
        / "predictions"
        / "mobilenet"
        / "lineup_segments.csv"
    ),
    "video_dir": PROJECT_ROOT / "data" / "raw_videos",
    "frames_dir": PROJECT_ROOT / "data" / "ocr_frames",
    "selected_frames_dir": (
        PROJECT_ROOT / "data" / "ocr_selected_frames"
    ),
    "frames_csv": OCR_OUTPUT_DIR / "ocr_frames.csv",
    "scout_output_csv": OCR_OUTPUT_DIR / "ocr_scout_detections.csv",
    "selected_frames_csv": (
        OCR_OUTPUT_DIR / "ocr_selected_frames.csv"
    ),
    "selection_diagnostics_csv": (
        OCR_OUTPUT_DIR / "ocr_frame_selection_diagnostics.csv"
    ),
    "output_csv": OCR_OUTPUT_DIR / "ocr_raw_detections.csv",
    "resolved_output_csv": OCR_OUTPUT_DIR / "resolved_lineups.csv",
    "resolved_diagnostics_csv": (
        OCR_OUTPUT_DIR / "resolved_lineups_diagnostics.csv"
    ),
    "attempts_csv": OCR_OUTPUT_DIR / "pipeline_attempts.csv",
    "cache_dir": PROJECT_ROOT / ".cache" / "paddlex",
}


@dataclass(frozen=True)
class PipelineConfig:
    segments_csv: Path
    video_dir: Path
    frames_dir: Path
    selected_frames_dir: Path
    frames_csv: Path
    scout_output_csv: Path
    selected_frames_csv: Path
    selection_diagnostics_csv: Path
    output_csv: Path
    resolved_output_csv: Path
    resolved_diagnostics_csv: Path
    attempts_csv: Path
    cache_dir: Path
    fps: float = 2.0
    scout_fps: float = 0.5
    jpeg_quality: int = 95
    min_score: float = 0.80
    ocr_batch_size: int = 8
    initial_frame_count: int = 3
    expanded_frame_count: int = 7
    players_per_lineup: int = 11
    min_number_count: int = 8
    same_lineup_gap_seconds: float = 20.0
    signature_threshold: float = 0.45
    min_pair_confidence: float = 0.80
    disable_local_ocr: bool = False
    extract_only: bool = False

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> PipelineConfig:
        values = {
            field.name: getattr(args, field.name)
            for field in fields(cls)
        }
        for name in DEFAULTS:
            values[name] = resolve_project_path(values[name])
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if self.fps <= 0 or self.scout_fps <= 0:
            raise LineupOCRError("--fps and --scout-fps must be positive.")
        if self.scout_fps > self.fps:
            raise LineupOCRError("--scout-fps cannot be greater than --fps.")
        if not 1 <= self.jpeg_quality <= 100:
            raise LineupOCRError("--jpeg-quality must be between 1 and 100.")
        if not 0 <= self.min_score <= 1:
            raise LineupOCRError("--min-score must be between 0 and 1.")
        if self.ocr_batch_size <= 0:
            raise LineupOCRError("--ocr-batch-size must be positive.")
        if self.initial_frame_count <= 0:
            raise LineupOCRError("--initial-frame-count must be positive.")
        if self.expanded_frame_count <= self.initial_frame_count:
            raise LineupOCRError(
                "--expanded-frame-count must exceed --initial-frame-count."
            )
        if self.players_per_lineup <= 0 or self.min_number_count <= 0:
            raise LineupOCRError(
                "--players-per-lineup and --min-number-count must be positive."
            )
        if self.same_lineup_gap_seconds < 0:
            raise LineupOCRError(
                "--same-lineup-gap-seconds cannot be negative."
            )
        if not 0 <= self.signature_threshold <= 1:
            raise LineupOCRError(
                "--signature-threshold must be between 0 and 1."
            )
        if not 0 <= self.min_pair_confidence <= 1:
            raise LineupOCRError(
                "--min-pair-confidence must be between 0 and 1."
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run adaptive lineup OCR: 3 selected frames, then 7, then "
            "the complete 2 FPS segment when the quality gate fails."
        )
    )
    for name in (
        "segments_csv",
        "video_dir",
        "frames_dir",
        "selected_frames_dir",
        "frames_csv",
        "scout_output_csv",
        "selected_frames_csv",
        "selection_diagnostics_csv",
        "output_csv",
        "resolved_output_csv",
        "resolved_diagnostics_csv",
        "attempts_csv",
        "cache_dir",
    ):
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=Path,
            default=DEFAULTS[name],
        )
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--scout-fps", type=float, default=0.5)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--min-score", type=float, default=0.80)
    parser.add_argument("--ocr-batch-size", type=int, default=8)
    parser.add_argument("--initial-frame-count", type=int, default=3)
    parser.add_argument("--expanded-frame-count", type=int, default=7)
    parser.add_argument("--players-per-lineup", type=int, default=11)
    parser.add_argument("--min-number-count", type=int, default=8)
    parser.add_argument("--same-lineup-gap-seconds", type=float, default=20.0)
    parser.add_argument("--signature-threshold", type=float, default=0.45)
    parser.add_argument("--min-pair-confidence", type=float, default=0.80)
    parser.add_argument(
        "--disable-local-ocr",
        action="store_true",
        help="Disable targeted OCR refinement inside the resolver.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Extract the 2 FPS frame metadata and stop before OCR.",
    )
    return parser


def parse_config() -> PipelineConfig:
    return PipelineConfig.from_namespace(build_parser().parse_args())
