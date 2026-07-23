"""Run a small PaddleOCR check on one lineup frame."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "ocr_samples"
    / "premier_match_01"
    / "liverpool"
    / "frames"
    / "premier_match_01"
    / "frame_000161.jpg"
)
DEFAULT_CACHE = PROJECT_ROOT / ".cache" / "paddlex"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PP-OCRv6 small models on one lineup frame."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Image to process.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE,
        help="Local PaddleX model cache.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.80,
        help="Only print text at or above this confidence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy frame: {input_path}")
    if not 0.0 <= args.min_score <= 1.0:
        raise ValueError("--min-score phải nằm trong khoảng 0..1")

    # This must be configured before importing PaddleOCR/PaddleX.
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    try:
        from paddleocr import PaddleOCR
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Chưa có PaddleOCR. Hãy chạy bằng `.venv-ocr/bin/python`."
        ) from exc

    started_at = time.perf_counter()
    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv6_small_det",
        text_recognition_model_name="PP-OCRv6_small_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
    )

    detections: list[dict[str, object]] = []
    for result in ocr.predict(str(input_path)):
        payload = result.json
        data = payload.get("res", payload)
        for text, score, box in zip(
            data.get("rec_texts", []),
            data.get("rec_scores", []),
            data.get("rec_boxes", []),
        ):
            score = float(score)
            if score >= args.min_score:
                detections.append(
                    {
                        "text": text,
                        "score": round(score, 4),
                        "box": list(map(int, box)),
                    }
                )

    print(
        json.dumps(
            {
                "input": str(input_path),
                "model_cache": str(cache_dir),
                "elapsed_seconds": round(time.perf_counter() - started_at, 3),
                "detections": detections,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
