"""PaddleOCR model setup and batched frame recognition."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .frames import FRAME_COLUMNS, LineupOCRError, resolve_project_path


DETECTION_MODEL = "PP-OCRv6_small_det"
RECOGNITION_MODEL = "PP-OCRv6_small_rec"
DETECTION_COLUMNS = FRAME_COLUMNS + [
    "text",
    "text_type",
    "score",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "center_x_norm",
    "center_y_norm",
]


def write_csv(
    rows: list[dict[str, object]],
    output_csv: Path,
    columns: list[str],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(output_csv, index=False)


def classify_text(text: str) -> str:
    if text.isdigit() and 1 <= int(text) <= 99:
        return "shirt_number_candidate"
    return "text"


def create_ocr(cache_dir: Path, recognition_batch_size: int):
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    try:
        from paddleocr import PaddleOCR
    except ModuleNotFoundError as exc:
        raise LineupOCRError(
            "PaddleOCR is not installed. Run this script with "
            "`.venv-ocr/bin/python`."
        ) from exc

    return PaddleOCR(
        text_detection_model_name=DETECTION_MODEL,
        text_recognition_model_name=RECOGNITION_MODEL,
        text_recognition_batch_size=recognition_batch_size,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
    )


def result_data(result: object) -> dict[str, object]:
    payload = getattr(result, "json", None)
    if callable(payload):
        payload = payload()
    if not isinstance(payload, dict):
        raise LineupOCRError("PaddleOCR returned an unsupported result format.")
    data = payload.get("res", payload)
    if not isinstance(data, dict):
        raise LineupOCRError(
            "PaddleOCR result does not contain a result object."
        )
    return data


def detections_from_result(
    frame_record: dict[str, object],
    result: object,
    min_score: float,
) -> list[dict[str, object]]:
    data = result_data(result)
    texts = data.get("rec_texts", [])
    scores = data.get("rec_scores", [])
    boxes = data.get("rec_boxes", [])
    if (
        not isinstance(texts, list)
        or len(texts) != len(scores)
        or len(texts) != len(boxes)
    ):
        raise LineupOCRError(
            "PaddleOCR returned inconsistent detection arrays."
        )

    frame_width = int(frame_record["frame_width"])
    frame_height = int(frame_record["frame_height"])
    detections: list[dict[str, object]] = []
    for text, score, raw_box in zip(texts, scores, boxes, strict=True):
        text = str(text).strip()
        score = float(score)
        if not text or score < min_score:
            continue
        try:
            box = [int(value) for value in raw_box]
        except (TypeError, ValueError) as exc:
            raise LineupOCRError(
                f"Unsupported OCR box: {raw_box}"
            ) from exc
        if len(box) != 4:
            raise LineupOCRError(f"Unsupported OCR box: {raw_box}")
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        detections.append(
            {
                **frame_record,
                "text": text,
                "text_type": classify_text(text),
                "score": round(score, 6),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": round(center_x, 3),
                "center_y": round(center_y, 3),
                "center_x_norm": round(center_x / frame_width, 6),
                "center_y_norm": round(center_y / frame_height, 6),
            }
        )
    return detections


def run_ocr(
    frame_records: list[dict[str, object]],
    cache_dir: Path,
    output_csv: Path | None,
    min_score: float,
    batch_size: int,
    ocr: object | None = None,
    progress_label: str = "OCR",
) -> list[dict[str, object]]:
    if ocr is None:
        print(f"Loading OCR models from: {cache_dir}")
        ocr = create_ocr(cache_dir, recognition_batch_size=batch_size)
    detections: list[dict[str, object]] = []

    for start in range(0, len(frame_records), batch_size):
        batch = frame_records[start : start + batch_size]
        input_paths = [
            str(resolve_project_path(Path(str(record["frame_path"]))))
            for record in batch
        ]
        results = list(ocr.predict(input_paths))
        if len(results) != len(batch):
            raise LineupOCRError(
                f"PaddleOCR returned {len(results)} result(s) for "
                f"{len(batch)} input frame(s)."
            )

        for frame_record, result in zip(batch, results, strict=True):
            detections.extend(
                detections_from_result(
                    frame_record,
                    result,
                    min_score=min_score,
                )
            )

        completed = min(start + len(batch), len(frame_records))
        if output_csv is not None:
            write_csv(detections, output_csv, DETECTION_COLUMNS)
        print(
            f"{progress_label}: {completed}/{len(frame_records)} frames, "
            f"{len(detections)} detections"
        )

    return detections
