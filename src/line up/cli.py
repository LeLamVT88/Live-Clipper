from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from line_up.config import PipelineConfig
from line_up.pipeline import detect_lineups


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


def main():
    parser = argparse.ArgumentParser(description="Lineup Test CLI: Detect football starting lineups from broadcast feeds.")
    parser.add_argument("video_path", type=str, help="Path to the input football video file")
    parser.add_argument("--max-scan", type=float, default=600.0, help="Maximum seconds to scan from start of video (default: 600s)")
    parser.add_argument("--output-json", type=str, default=None, help="Path to write JSON detection results")
    parser.add_argument("--export-clips-dir", type=str, default=None, help="Directory to export cut lineup video clips")
    parser.add_argument("--extract-starters", action="store_true", help="Select graphic windows and OCR starters after detection")

    args = parser.parse_args()
    cfg = PipelineConfig(max_scan_seconds=args.max_scan)

    print(f"Running lineup detection on: {args.video_path}")
    result = detect_lineups(args.video_path, config=cfg)
    output = result.to_dict()
    if args.extract_starters:
        from mapping.pipeline import extract_lineup_graphics
        output["graphic_extractions"] = [
            extract_lineup_graphics(args.video_path, interval.start_seconds, interval.end_seconds)
            for interval in result.lineups
        ]
        for extraction in output["graphic_extractions"]:
            stats = extraction["stats"]
            print(f"Graphic selection: {stats['status']} ({stats['total_seconds']:.2f}s including extraction)")
            for graphic in extraction["graphics"]:
                pairs = sum(p["confirmed"] for p in graphic["players"])
                print(f"  {graphic['layout']}: {graphic['status']}, {pairs}/11 confirmed pairs")

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

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nJSON results saved to: {out_json}")

    if args.export_clips_dir and result.lineups:
        export_dir = Path(args.export_clips_dir)
        vpath = Path(args.video_path)
        print(f"\nExporting {len(result.lineups)} clips to: {export_dir}...")
        for idx, lineup in enumerate(result.lineups, 1):
            clip_name = f"{vpath.stem}_lineup_{idx}.mp4"
            clip_path = export_dir / clip_name
            ok = export_video_clip(vpath, lineup.start_seconds, lineup.end_seconds, clip_path)
            if ok:
                print(f"  Exported: {clip_path}")


if __name__ == "__main__":
    main()
