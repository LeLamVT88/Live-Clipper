from __future__ import annotations

import argparse
import json
from pathlib import Path

from mapping.csv_export import export_results
from mapping.pipeline import MappingConfig, extract_lineup_graphics


def _resolve_video(path: Path) -> Path:
    if path.exists():
        return path
    matches = list((Path.cwd() / "data").rglob(path.name))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Video from detection JSON no longer exists: {path}")
    raise ValueError(f"Ambiguous relocated video {path.name}: {matches}")


def _load_jobs(args: argparse.Namespace) -> list[tuple[Path, float, float]]:
    if args.detection_json:
        data = json.loads(args.detection_json.read_text(encoding="utf-8"))
        video = _resolve_video(Path(data["video_path"]))
        return [(video, float(item["start_seconds"]), float(item["end_seconds"]))
                for item in data.get("lineups", [])]
    if args.video_path is None or args.start is None or args.end is None:
        raise ValueError("Provide VIDEO_PATH with --start/--end, or use --detection-json")
    return [(_resolve_video(args.video_path), args.start, args.end)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract only starting-player name/number pairs from lineup graphics")
    parser.add_argument("video_path", nargs="?", type=Path)
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    parser.add_argument("--detection-json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scout-fps", type=float, default=2.0)
    parser.add_argument("--fallback-scout-fps", type=float, default=5.0)
    args = parser.parse_args()

    jobs = _load_jobs(args)
    config = MappingConfig(
        scout_fps=args.scout_fps,
        fallback_scout_fps=max(args.scout_fps, args.fallback_scout_fps),
    )
    extractions = []
    for video, start, end in jobs:
        print(f"Mapping starters: {video} [{start:.3f}, {end:.3f}]", flush=True)
        extraction = extract_lineup_graphics(video, start, end, config)
        extractions.append(extraction)
        print(f"  {extraction['stats']['status']}: {len(extraction['graphics'])} stable graphic(s)", flush=True)
        for graphic in extraction["graphics"]:
            confirmed = sum(bool(player["confirmed"]) for player in graphic["players"])
            issue_text = ", ".join(graphic["issues"]) or "none"
            print(
                f"    graphic {graphic['graphic_index']}: {graphic['status']} "
                f"({confirmed}/11 confirmed; issues: {issue_text})",
                flush=True,
            )
    paths = export_results(extractions, args.output_dir)
    print(f"Resolved CSV: {paths['resolved']}")
    print(f"Partial CSV: {paths['partial']}")
    print(f"Diagnostics: {paths['diagnostics']}")
    print(f"Evidence: {paths['evidence']}")


if __name__ == "__main__":
    main()
