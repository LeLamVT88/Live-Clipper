from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import cv2

from .frames import relative_to_project, resolve_project_path
from .selector import LineupFrameSelectionError, SegmentSelection


def materialize_selected_frames(
    frame_records: list[dict[str, object]], selections: list[SegmentSelection],
    output_dir: Path, jpeg_quality: int,
) -> list[dict[str, object]]:
    records_by_key = {
        (
            str(record["video"]),
            int(record["segment_index"]),
            int(record["frame_index"]),
        ): record
        for record in frame_records
    }
    selected_records: list[dict[str, object]] = []
    for selection in selections:
        for frame_index in selection.selected_frame_indices:
            key = (selection.video, selection.segment_index, frame_index)
            source_record = records_by_key.get(key)
            if source_record is None:
                raise LineupFrameSelectionError(f"Selected frame metadata is missing: {key}")
            source_path = resolve_project_path(Path(str(source_record["frame_path"])))
            if not source_path.is_file():
                raise LineupFrameSelectionError(
                    f"Selected source frame does not exist: {source_path}"
                )
            output_record = dict(source_record)
            output_record.update(
                {
                    "source_frame_path": relative_to_project(source_path),
                    "selection_layout": selection.layout, "selection_score": round(selection.score, 3),
                    "scout_frame_index": selection.scout_frame_index, "crop_side": selection.crop_side,
                    "crop_x1_norm": selection.crop_x1_norm, "crop_x2_norm": selection.crop_x2_norm,
                }
            )
            if selection.crop_side == "full":
                selected_records.append(output_record)
                continue
            image = cv2.imread(str(source_path))
            if image is None:
                raise LineupFrameSelectionError(f"Cannot read selected source frame: {source_path}")
            width = image.shape[1]
            x1 = max(0, min(width - 1, round(selection.crop_x1_norm * width)))
            x2 = max(x1 + 1, min(width, round(selection.crop_x2_norm * width)))
            crop = image[:, x1:x2]
            segment_dir = output_dir / Path(str(source_record["frame_path"])).parent.parent.name
            segment_dir /= f"segment_{selection.segment_index:02d}"
            segment_dir.mkdir(parents=True, exist_ok=True)
            timestamp_ms = round(float(source_record["timestamp_seconds"]) * 1000)
            crop_path = segment_dir / f"frame_{frame_index:06d}_t{timestamp_ms:010d}.jpg"
            if not cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
                raise LineupFrameSelectionError(f"Cannot write selected crop: {crop_path}")
            crop_height, crop_width = crop.shape[:2]
            output_record["frame_path"] = relative_to_project(crop_path)
            output_record["frame_width"] = crop_width
            output_record["frame_height"] = crop_height
            selected_records.append(output_record)
    return sorted(
        selected_records,
        key=lambda row: (row["video"], row["segment_index"], row["timestamp_seconds"]),
    )


def selection_diagnostic_rows(selections: list[SegmentSelection]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for selection in selections:
        row = asdict(selection)
        row.update(
            score=round(selection.score, 3),
            crop_side=selection.crop_side,
            selected_frame_indices="|".join(map(str, selection.selected_frame_indices)),
        )
        rows.append(row)
    return rows
