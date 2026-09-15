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
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
MAPPING_CSV_FIELDS = (
    "frames_csv",
    "scout_output_csv",
    "selected_frames_csv",
    "selection_diagnostics_csv",
    "output_csv",
    "resolved_output_csv",
    "resolved_diagnostics_csv",
    "attempts_csv",
)


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


def default_output_dir(video_path: Path) -> Path:
    resolved = video_path.expanduser().resolve()
    try:
        relative = resolved.relative_to(DATA_ROOT).with_suffix("")
    except ValueError:
        return OUTPUT_ROOT / safe_stem(resolved.stem)
    return OUTPUT_ROOT.joinpath(*(safe_stem(part) for part in relative.parts))


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


def _json_shirt_number(value: str) -> int | str:
    normalized = value.strip()
    try:
        return int(normalized)
    except ValueError:
        return normalized


def write_players_json(resolved_csv: Path | None, output_json: Path) -> int:
    rows: list[dict[str, object]] = []
    if resolved_csv is not None and resolved_csv.is_file():
        with resolved_csv.open(newline="", encoding="utf-8-sig") as source:
            for row in csv.DictReader(source):
                rows.append(
                    {
                        "lineup_clip": Path(row.get("video", "")).name,
                        "lineup_index": int(row.get("lineup_index") or 1),
                        "shirt_number": _json_shirt_number(row.get("shirt_number", "")),
                        "player_name": row.get("player_name", ""),
                    }
                )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as destination:
        json.dump(rows, destination, indent=2, ensure_ascii=False)
        destination.write("\n")
    return len(rows)


def remove_intermediate_csvs(config: MappingConfig) -> None:
    for field_name in MAPPING_CSV_FIELDS:
        getattr(config, field_name).unlink(missing_ok=True)


def remove_legacy_players_csv(output_dir: Path) -> None:
    (output_dir / "players.csv").unlink(missing_ok=True)


def run_full_pipeline(
    video_path: Path, output_dir: Path, max_scan: float | None = None,
    disable_local_ocr: bool = False, keep_diagnostics: bool = False,
) -> int:
    video_path = video_path.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")
    if max_scan is not None and max_scan <= 0:
        raise ValueError("--max-scan must be positive.")
    output_dir = output_dir.expanduser().resolve()
    clips_dir = output_dir / "clips"
    detection_json = output_dir / "detection_result.json"
    players_json = output_dir / "players.json"
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

    # A rerun can find fewer intervals than an earlier run. Remove only clips
    # generated for this source video so stale false positives are not left in
    # the result directory.
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_prefix = f"{safe_stem(video_path.stem)}_lineup_"
    for stale_clip in clips_dir.glob(f"{clip_prefix}*.mp4"):
        stale_clip.unlink()

    if not result.lineups:
        write_players_json(None, players_json)
        remove_legacy_players_csv(output_dir)
        print(f"No lineup interval found. Detection details: {detection_json}")
        return 2

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

    player_count = write_players_json(config.resolved_output_csv, players_json)
    remove_legacy_players_csv(output_dir)
    if not keep_diagnostics:
        remove_intermediate_csvs(config)
    print("\n" + "=" * 60)
    print(f"FULL PIPELINE COMPLETE: {player_count} player row(s)")
    print(f"Lineup clips : {clips_dir}")
    print(f"Players JSON : {players_json}")
    if keep_diagnostics:
        print(f"Detailed CSV : {config.resolved_output_csv}")
    print(f"Detection JSON: {detection_json}")
    return mapping_status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect lineup graphics in one football match, export the lineup clips, "
            "then write player names and shirt numbers to JSON."
        )
    )
    parser.add_argument("video_path", type=Path, help="Input football match video")
    parser.add_argument(
        "--output-dir", type=Path,
        help="Output directory (default for data/<league>/<match>: outputs/<league>/<match>)",
    )
    parser.add_argument(
        "--max-scan", type=float,
        help="Optionally limit scanning to the first N seconds (default: full video)",
    )
    parser.add_argument(
        "--disable-local-ocr", action="store_true",
        help="Disable targeted OCR refinement in the player mapper.",
    )
    parser.add_argument(
        "--keep-diagnostics", action="store_true",
        help="Keep intermediate mapping CSV reports for debugging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or default_output_dir(args.video_path)
    try:
        return run_full_pipeline(
            args.video_path, output_dir, args.max_scan, args.disable_local_ocr,
            args.keep_diagnostics,
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
