from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path

from line_up.config import PipelineConfig
from line_up.pipeline import detect_lineups
from mapping.config import DEFAULTS as MAPPING_DEFAULTS
from mapping.config import PipelineConfig as MappingConfig
from mapping.frames import LineupOCRError, safe_stem
from mapping.pipeline import LineupWorkflow
from mapping.resolver import LineupResolutionError
from mapping.selector import LineupFrameSelectionError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLAYER_COLUMNS = ("lineup_clip", "shirt_number", "player_name")


def export_video_clip(video_path: Path, start: float, end: float, output_path: Path) -> bool:
    """Export precise video segment without re-encoding using ffmpeg stream copy."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end - start
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", f"{start:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-c", "copy",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        # Fallback to re-encoding if copy fails at keyframe boundary
        fallback_cmd = [
            "ffmpeg",
            "-y",
            "-ss", f"{start:.3f}",
            "-i", str(video_path),
            "-t", f"{duration:.3f}",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-c:a", "aac",
            str(output_path),
        ]
        try:
            subprocess.run(fallback_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except Exception as err:
            print(f"Error exporting clip to {output_path}: {err}")
            return False


def mapping_config(clips_dir: Path, output_dir: Path, disable_local_ocr: bool) -> MappingConfig:
    artifacts = output_dir / "mapping"
    config = MappingConfig(
        clips_dir=clips_dir,
        frames_dir=artifacts / "frames",
        selected_frames_dir=artifacts / "selected_frames",
        frames_csv=artifacts / "ocr_frames.csv",
        scout_output_csv=artifacts / "ocr_scout_detections.csv",
        selected_frames_csv=artifacts / "ocr_selected_frames.csv",
        selection_diagnostics_csv=artifacts / "ocr_frame_selection_diagnostics.csv",
        output_csv=artifacts / "ocr_raw_detections.csv",
        resolved_output_csv=output_dir / "resolved_lineups.csv",
        resolved_diagnostics_csv=artifacts / "resolved_lineups_diagnostics.csv",
        attempts_csv=artifacts / "pipeline_attempts.csv",
        cache_dir=Path(MAPPING_DEFAULTS["cache_dir"]),
        disable_local_ocr=disable_local_ocr,
    )
    config.validate()
    return config


def write_players_csv(resolved_csv: Path | None, output_csv: Path) -> int:
    rows: list[dict[str, object]] = []
    if resolved_csv is not None and resolved_csv.is_file():
        with resolved_csv.open(newline="", encoding="utf-8-sig") as source:
            for row in csv.DictReader(source):
                rows.append(
                    {
                        "lineup_clip": Path(row.get("video", "")).name,
                        "shirt_number": row.get("shirt_number", ""),
                        "player_name": row.get("player_name", ""),
                    }
                )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=PLAYER_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def run_full_pipeline(
    video_path: Path, output_dir: Path, max_scan: float = 600.0,
    disable_local_ocr: bool = False,
) -> int:
    video_path = video_path.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")
    if max_scan <= 0:
        raise ValueError("--max-scan must be positive.")
    output_dir = output_dir.expanduser().resolve()
    clips_dir = output_dir / "clips"
    detection_json = output_dir / "detection_result.json"
    players_csv = output_dir / "players.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running lineup detection on: {video_path}")
    result = detect_lineups(video_path, config=PipelineConfig(max_scan_seconds=max_scan))
    output = result.to_dict()
    print("\n" + "=" * 60)
    print(f"DETECTION COMPLETE ({result.processing_time_seconds:.2f}s total)")
    print("=" * 60)
    print(f"Scenes detected : {len(result.scenes)}")
    print(f"Frames evaluated: {len(result.sampled_frames)}")
    print(f"Lineups found   : {len(result.lineups)}")

    for idx, lineup in enumerate(result.lineups, 1):
        merged_str = " [Merged Back-to-Back]" if lineup.metadata.get("merged_adjacent") else ""
        print(f"  [{idx}] {lineup.start_seconds:.2f}s -> {lineup.end_seconds:.2f}s "
              f"(duration: {lineup.duration:.2f}s, confidence: {lineup.confidence:.3f}){merged_str}")

    with detection_json.open("w", encoding="utf-8") as destination:
        json.dump(output, destination, indent=2, ensure_ascii=False)
    if not result.lineups:
        write_players_csv(None, players_csv)
        print(f"No lineup interval found. Detection details: {detection_json}")
        return 2

    clips_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nExporting {len(result.lineups)} clip(s) to: {clips_dir}")
    clips: list[Path] = []
    for index, lineup in enumerate(result.lineups, 1):
        clip_path = clips_dir / f"{safe_stem(video_path.stem)}_lineup_{index}.mp4"
        if not export_video_clip(video_path, lineup.start_seconds, lineup.end_seconds, clip_path):
            raise RuntimeError(f"Could not export lineup clip: {clip_path}")
        clips.append(clip_path)
        print(f"  Exported: {clip_path}")

    # Give mapping a clean manifest directory so stale clips from an older run
    # in output_dir/clips can never leak into the current match result.
    with tempfile.TemporaryDirectory(prefix="lineup-mapping-") as temporary:
        mapping_input = Path(temporary)
        for clip in clips:
            (mapping_input / clip.name).symlink_to(clip)
        config = mapping_config(mapping_input, output_dir, disable_local_ocr)
        mapping_status = LineupWorkflow(config).run()

    player_count = write_players_csv(config.resolved_output_csv, players_csv)
    print("\n" + "=" * 60)
    print(f"FULL PIPELINE COMPLETE: {player_count} player row(s)")
    print(f"Lineup clips : {clips_dir}")
    print(f"Players CSV  : {players_csv}")
    print(f"Detailed CSV : {config.resolved_output_csv}")
    print(f"Detection JSON: {detection_json}")
    return mapping_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect lineup graphics in one football match, export the lineup clips, "
            "then write player names and shirt numbers to CSV."
        )
    )
    parser.add_argument("video_path", type=Path, help="Input football match video")
    parser.add_argument(
        "--output-dir", type=Path,
        help="Output directory (default: outputs/<video-name>)",
    )
    parser.add_argument(
        "--max-scan", type=float, default=600.0,
        help="Maximum seconds scanned from the start of the match (default: 600)",
    )
    parser.add_argument(
        "--disable-local-ocr", action="store_true",
        help="Disable targeted OCR refinement in the player mapper.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or PROJECT_ROOT / "outputs" / safe_stem(args.video_path.stem)
    try:
        return run_full_pipeline(
            args.video_path, output_dir, args.max_scan, args.disable_local_ocr
        )
    except (
        FileNotFoundError,
        LineupFrameSelectionError,
        LineupOCRError,
        LineupResolutionError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"Full lineup pipeline failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
