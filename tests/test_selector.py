from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mapping.selector as selector
from mapping.selection_io import materialize_selected_frames


def frame_record(
    frame_index: int,
    timestamp_seconds: float,
    frame_path: str = "frame.jpg",
) -> dict[str, object]:
    return {
        "video": "match.mp4",
        "segment_index": 1,
        "segment_label": "lineup_01",
        "segment_start_seconds": 0.0,
        "segment_end_seconds": 4.0,
        "frame_index": frame_index,
        "frame_path": frame_path,
        "timestamp": f"00:00:0{timestamp_seconds:g}",
        "timestamp_seconds": timestamp_seconds,
        "relative_seconds": timestamp_seconds,
        "frame_width": 200,
        "frame_height": 100,
    }


def detection(
    *,
    frame_index: int,
    timestamp_seconds: float,
    text: str,
    text_type: str,
    x: float,
    y: float,
) -> dict[str, object]:
    return {
        **frame_record(frame_index, timestamp_seconds),
        "text": text,
        "text_type": text_type,
        "score": 0.99,
        "x1": int((x - 0.02) * 200),
        "y1": int((y - 0.02) * 100),
        "x2": int((x + 0.02) * 200),
        "y2": int((y + 0.02) * 100),
        "center_x": x * 200,
        "center_y": y * 100,
        "center_x_norm": x,
        "center_y_norm": y,
    }


def left_substitutes_formation(
    frame_index: int,
    timestamp_seconds: float,
) -> list[dict[str, object]]:
    rows = [
        detection(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            text="SUBSTITUTES",
            text_type="text",
            x=0.15,
            y=0.15,
        )
    ]
    for number, name, x, y in [
        ("7", "PLAYER A", 0.55, 0.25),
        ("10", "PLAYER B", 0.72, 0.40),
        ("3", "PLAYER C", 0.62, 0.60),
    ]:
        rows.extend(
            [
                detection(
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    text=number,
                    text_type="shirt_number_candidate",
                    x=x,
                    y=y,
                ),
                detection(
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    text=name,
                    text_type="text",
                    x=x,
                    y=y + 0.06,
                ),
            ]
        )
    return rows


class ScoutSamplingTests(unittest.TestCase):
    def test_samples_sparse_frames_per_segment(self) -> None:
        records = [frame_record(index, (index - 1) * 0.5) for index in range(1, 9)]

        sampled = selector.sample_scout_frames(records, scout_fps=0.5)

        self.assertEqual(
            [int(record["frame_index"]) for record in sampled],
            [1, 5],
        )


class SegmentSelectionTests(unittest.TestCase):
    def test_selects_three_frames_and_crops_opposite_substitutes(self) -> None:
        records = [frame_record(index, (index - 1) * 0.5) for index in range(1, 9)]
        detections = pd.DataFrame(left_substitutes_formation(5, 2.0))

        selections = selector.select_segment_frames(
            records,
            detections,
            selected_frame_count=3,
            scout_fps=0.5,
        )

        self.assertEqual(len(selections), 1)
        selection = selections[0]
        self.assertEqual(selection.status, "selected")
        self.assertEqual(selection.layout, "formation_substitutes_left")
        self.assertEqual(selection.crop_side, "right")
        self.assertGreater(selection.crop_x1_norm, 0.25)
        self.assertEqual(selection.selected_frame_indices, (4, 5, 6))

    def test_falls_back_to_all_frames_without_candidate(self) -> None:
        records = [frame_record(index, (index - 1) * 0.5) for index in range(1, 5)]
        detections = pd.DataFrame(
            [
                detection(
                    frame_index=1,
                    timestamp_seconds=0.0,
                    text="LIVE",
                    text_type="text",
                    x=0.9,
                    y=0.1,
                )
            ]
        )

        selections = selector.select_segment_frames(
            records,
            detections,
            selected_frame_count=3,
            scout_fps=0.5,
        )

        self.assertEqual(selections[0].status, "fallback")
        self.assertEqual(selections[0].selected_frame_indices, (1, 2, 3, 4))

    def test_spreads_frames_across_stable_formation_scene(self) -> None:
        records = [frame_record(index, (index - 1) * 0.5) for index in range(1, 14)]
        candidates = [
            selector.FrameCandidate(
                video="match.mp4",
                segment_index=1,
                frame_index=frame_index,
                timestamp_seconds=timestamp,
                layout="formation_substitutes_left",
                score=50.0,
                formation_anchor_count=5,
                table_pair_count=0,
                number_count=5,
                name_count=11,
                crop_x1_norm=0.35,
                crop_x2_norm=1.0,
            )
            for frame_index, timestamp in [(1, 0.0), (5, 2.0), (9, 4.0), (13, 6.0)]
        ]

        selected = selector.choose_spread_frame_indices(
            records,
            candidates,
            best=candidates[1],
            count=3,
            scout_period_seconds=2.0,
        )

        self.assertEqual(selected, (1, 9, 13))

    def test_materializes_only_the_formation_side(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            source_path = temporary_path / "frame.jpg"
            image = np.zeros((100, 200, 3), dtype=np.uint8)
            image[:, :70] = (255, 0, 0)
            image[:, 70:] = (0, 255, 0)
            self.assertTrue(cv2.imwrite(str(source_path), image))

            record = frame_record(1, 0.0, frame_path=str(source_path))
            selection = selector.SegmentSelection(
                video="match.mp4",
                segment_index=1,
                layout="formation_substitutes_left",
                status="selected",
                score=50.0,
                scout_frame_index=1,
                scout_timestamp_seconds=0.0,
                formation_anchor_count=3,
                table_pair_count=0,
                number_count=3,
                name_count=3,
                crop_x1_norm=0.35,
                crop_x2_norm=1.0,
                selected_frame_indices=(1,),
                message="selected",
            )

            selected = materialize_selected_frames(
                [record],
                [selection],
                output_dir=temporary_path / "selected",
                jpeg_quality=95,
            )

            crop_path = Path(str(selected[0]["frame_path"]))
            if not crop_path.is_absolute():
                crop_path = PROJECT_ROOT / crop_path
            crop = cv2.imread(str(crop_path))
            self.assertIsNotNone(crop)
            assert crop is not None
            self.assertEqual(crop.shape[:2], (100, 130))
            self.assertEqual(selected[0]["crop_side"], "right")


if __name__ == "__main__":
    unittest.main()
