from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path

from .frames import PROJECT_ROOT, LineupOCRError, resolve_project_path


OCR_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "predictions" / "ocr"
DEFAULTS = {
    "clips_dir": PROJECT_ROOT / "outputs" / "clips",
    "frames_dir": PROJECT_ROOT / "data" / "ocr_frames",
    "selected_frames_dir": (PROJECT_ROOT / "data" / "ocr_selected_frames"),
    "frames_csv": OCR_OUTPUT_DIR / "ocr_frames.csv",
    "scout_output_csv": OCR_OUTPUT_DIR / "ocr_scout_detections.csv",
    "selected_frames_csv": (OCR_OUTPUT_DIR / "ocr_selected_frames.csv"),
    "selection_diagnostics_csv": (OCR_OUTPUT_DIR / "ocr_frame_selection_diagnostics.csv"),
    "output_csv": OCR_OUTPUT_DIR / "ocr_raw_detections.csv",
    "resolved_output_csv": OCR_OUTPUT_DIR / "resolved_lineups.csv",
    "resolved_diagnostics_csv": (OCR_OUTPUT_DIR / "resolved_lineups_diagnostics.csv"),
    "attempts_csv": OCR_OUTPUT_DIR / "pipeline_attempts.csv",
    "cache_dir": PROJECT_ROOT / ".cache" / "paddlex",
}


@dataclass(frozen=True)
class PipelineConfig:
    clips_dir: Path
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
    lineups_per_match: int = 2
    min_number_count: int = 8
    same_lineup_gap_seconds: float = 20.0
    signature_threshold: float = 0.45
    min_pair_confidence: float = 0.80
    disable_local_ocr: bool = False
    extract_only: bool = False
    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> PipelineConfig:
        values = {field.name: getattr(args, field.name) for field in fields(cls)}
        for name in DEFAULTS:
            values[name] = resolve_project_path(values[name])
        config = cls(**values)
        config.validate()
        return config
    def validate(self) -> None:
        checks = (
            (self.fps > 0 and self.scout_fps > 0, "--fps and --scout-fps must be positive."),
            (self.scout_fps <= self.fps, "--scout-fps cannot be greater than --fps."),
            (1 <= self.jpeg_quality <= 100, "--jpeg-quality must be between 1 and 100."),
            (0 <= self.min_score <= 1, "--min-score must be between 0 and 1."),
            (self.ocr_batch_size > 0, "--ocr-batch-size must be positive."),
            (self.initial_frame_count > 0, "--initial-frame-count must be positive."),
            (self.expanded_frame_count > self.initial_frame_count,
             "--expanded-frame-count must exceed --initial-frame-count."),
            (self.players_per_lineup > 0 and self.lineups_per_match > 0
             and self.min_number_count > 0,
             "--players-per-lineup, --lineups-per-match, and --min-number-count "
             "must be positive."),
            (self.same_lineup_gap_seconds >= 0, "--same-lineup-gap-seconds cannot be negative."),
            (0 <= self.signature_threshold <= 1, "--signature-threshold must be between 0 and 1."),
            (0 <= self.min_pair_confidence <= 1, "--min-pair-confidence must be between 0 and 1."),
        )
        if message := next((message for valid, message in checks if not valid), None):
            raise LineupOCRError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read lineup clips exported by the PySceneDetect + OCR detector, "
            "then run adaptive OCR on 3 frames, 7 frames, or the complete "
            "2 FPS clip when the quality gate fails."
        )
    )
    for name in DEFAULTS:
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=DEFAULTS[name])
    numeric = {
        "fps": (float, 2.0), "scout-fps": (float, 0.5), "jpeg-quality": (int, 95),
        "min-score": (float, 0.80), "ocr-batch-size": (int, 8),
        "initial-frame-count": (int, 3), "expanded-frame-count": (int, 7),
        "players-per-lineup": (int, 11), "lineups-per-match": (int, 2),
        "min-number-count": (int, 8),
        "same-lineup-gap-seconds": (float, 20.0), "signature-threshold": (float, 0.45),
        "min-pair-confidence": (float, 0.80),
    }
    for name, (value_type, default) in numeric.items():
        parser.add_argument(f"--{name}", type=value_type, default=default)
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
