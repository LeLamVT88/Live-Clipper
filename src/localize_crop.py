from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

from utils import ensure_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "frame_labels.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "crops"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "crop_metadata.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop a fixed region from lineup frames for later OCR/localization checks."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--x", type=int, required=True)
    parser.add_argument("--y", type=int, required=True)
    parser.add_argument("--w", type=int, required=True)
    parser.add_argument("--h", type=int, required=True)
    parser.add_argument(
        "--only-lineup",
        action="store_true",
        help="Only crop rows where label is 1.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.w <= 0 or args.h <= 0:
        print("Crop width and height must be greater than 0.")
        return 1
    if not args.input_csv.exists():
        print(f"Input CSV does not exist: {args.input_csv}")
        return 1

    frames = pd.read_csv(args.input_csv)
    required_columns = {"video", "frame_path", "timestamp", "timestamp_seconds"}
    missing = sorted(required_columns - set(frames.columns))
    if missing:
        print(f"Missing columns in {args.input_csv}: {', '.join(missing)}")
        return 1

    if args.only_lineup:
        if "label" not in frames.columns:
            print("--only-lineup requires a label column.")
            return 1
        frames = frames[frames["label"] == 1]

    ensure_dir(args.output_dir)
    ensure_dir(args.output_csv.parent)
    records: list[dict[str, object]] = []
    errors = 0

    for index, row in frames.iterrows():
        frame_path = Path(str(row["frame_path"]))
        full_frame_path = frame_path if frame_path.is_absolute() else PROJECT_ROOT / frame_path
        image = cv2.imread(str(full_frame_path))
        if image is None:
            errors += 1
            continue

        crop = image[args.y : args.y + args.h, args.x : args.x + args.w]
        if crop.size == 0:
            errors += 1
            continue

        video_stem = Path(str(row["video"])).stem
        video_crop_dir = ensure_dir(args.output_dir / video_stem)
        crop_path = video_crop_dir / f"crop_{index + 1:06d}.jpg"
        cv2.imwrite(str(crop_path), crop)

        records.append(
            {
                "video": row["video"],
                "frame_path": row["frame_path"],
                "crop_path": crop_path.relative_to(PROJECT_ROOT).as_posix(),
                "timestamp": row["timestamp"],
                "timestamp_seconds": row["timestamp_seconds"],
                "x": args.x,
                "y": args.y,
                "w": args.w,
                "h": args.h,
            }
        )

    pd.DataFrame(records).to_csv(args.output_csv, index=False)
    print(f"Saved {len(records)} crop(s) to: {args.output_dir}")
    print(f"Saved crop metadata to: {args.output_csv}")
    if errors:
        print(f"Skipped {errors} frame(s) because they could not be read or cropped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
