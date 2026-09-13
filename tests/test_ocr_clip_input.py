from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mapping.config import PipelineConfig, build_parser
from mapping.frames import (
    LineupOCRError,
    Segment,
    extract_segment_frames,
    load_lineup_clips,
    video_duration_seconds,
)
from mapping.pipeline import LineupWorkflow


class OCRClipInputTests(unittest.TestCase):
    def test_cli_defaults_to_new_detector_output_clips(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.clips_dir, Path("outputs/clips").resolve())
        self.assertFalse(hasattr(args, "segments_csv"))
        self.assertFalse(hasattr(args, "video_dir"))

    def test_discovers_exported_clips_recursively(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a_lineup_1.mp4"
            second = root / "league" / "b_lineup_2.MOV"
            ignored = root / "detections.json"
            second.parent.mkdir()
            first.write_bytes(b"video")
            second.write_bytes(b"video")
            ignored.write_text("{}", encoding="utf-8")

            with patch("mapping.frames.video_duration_seconds", return_value=12.5):
                segments = load_lineup_clips(root)

        self.assertEqual(
            [segment.video_path.name for segment in segments],
            ["a_lineup_1.mp4", "b_lineup_2.MOV"],
        )
        self.assertTrue(all(segment.start_seconds == 0 for segment in segments))
        self.assertTrue(all(segment.end_seconds == 12.5 for segment in segments))
        self.assertTrue(all(segment.index == 1 for segment in segments))

    def test_rejects_directory_without_video_clips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(LineupOCRError, "No lineup video clips"):
                load_lineup_clips(Path(directory))

    def test_duration_prefers_ffprobe_for_stream_copy_clip(self) -> None:
        completed = SimpleNamespace(stdout="35.634000\n")

        with (
            patch("mapping.frames.subprocess.run", return_value=completed),
            patch("mapping.frames.cv2.VideoCapture") as capture,
        ):
            duration = video_duration_seconds(Path("lineup.mp4"))

        self.assertEqual(duration, 35.634)
        capture.assert_not_called()

    def test_frame_extraction_tolerates_undecodable_final_slot(self) -> None:
        class Capture:
            def __init__(self) -> None:
                self.timestamp = 0.0

            def isOpened(self) -> bool:
                return True

            def get(self, prop: int) -> float:
                if prop == 5:  # cv2.CAP_PROP_FPS
                    return 2.0
                if prop == 7:  # cv2.CAP_PROP_FRAME_COUNT
                    return 4.0
                return self.timestamp * 1000

            def set(self, prop: int, value: float) -> None:
                self.timestamp = value / 1000

            def read(self) -> tuple[bool, np.ndarray | None]:
                if self.timestamp >= 1.5:
                    return False, None
                return True, np.zeros((10, 20, 3), dtype=np.uint8)

            def release(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as directory:
            segment = Segment(
                video="outputs/clips/lineup.mp4",
                video_path=Path(directory) / "lineup.mp4",
                index=1,
                label="lineup",
                start_seconds=0.0,
                end_seconds=2.0,
            )
            with (
                patch("mapping.frames.cv2.VideoCapture", return_value=Capture()),
                patch("mapping.frames.cv2.imwrite", return_value=True),
            ):
                records = extract_segment_frames(
                    segment,
                    frames_dir=Path(directory) / "frames",
                    fps=2.0,
                    jpeg_quality=95,
                )

        self.assertEqual([row["timestamp_seconds"] for row in records], [0.0, 0.5, 1.0])

    def test_workflow_extracts_each_discovered_clip_as_a_full_segment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = PipelineConfig.from_namespace(
                build_parser().parse_args(["--clips-dir", str(root), "--extract-only"])
            )
            segment = SimpleNamespace(
                video_path=root / "match_lineup_1.mp4",
                index=1,
                start_seconds=0.0,
                end_seconds=8.0,
            )
            frame = {
                "video": "outputs/clips/match_lineup_1.mp4",
                "segment_index": 1,
                "frame_index": 1,
            }

            with (
                patch("mapping.pipeline.load_lineup_clips", return_value=[segment]) as load,
                patch("mapping.pipeline.extract_segment_frames", return_value=[frame]) as extract,
                patch("mapping.pipeline.write_csv") as write,
            ):
                result = LineupWorkflow(config)._extract_frames()

        self.assertEqual(result, [frame])
        load.assert_called_once_with(config.clips_dir)
        self.assertEqual(extract.call_args.args[0], segment)
        write.assert_called_once()


if __name__ == "__main__":
    unittest.main()
