from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lineup.export_clips import (
    ClipExportError,
    ClipJob,
    load_segments,
    validate_jobs,
)
from transcript.audio import MediaInfo


class ClipExportValidationTests(unittest.TestCase):
    def test_loads_coarse_lineup_csv_directly(self) -> None:
        csv_text = "video,start_seconds,end_seconds\nmatch.mp4,10,20\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "segments.csv"
            path.write_text(csv_text, encoding="utf-8")

            rows = load_segments(path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(float(rows.iloc[0]["_start_seconds"]), 10.0)
        self.assertEqual(float(rows.iloc[0]["_end_seconds"]), 20.0)

    def test_rejects_segment_beyond_source_duration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "match.mp4"
            source.write_bytes(b"video")
            job = ClipJob(
                source=source,
                output=root / "clip.mp4",
                start_seconds=90.0,
                end_seconds=101.0,
            )
            media = MediaInfo(
                duration_seconds=100.0,
                audio_stream_count=1,
                video_stream_count=1,
            )

            with patch("lineup.export_clips.probe_media", return_value=media):
                with self.assertRaisesRegex(ClipExportError, "beyond source"):
                    validate_jobs([job], overwrite=False)


if __name__ == "__main__":
    unittest.main()
