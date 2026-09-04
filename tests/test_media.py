from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lineup.media import probe_media


class MediaProbeTests(unittest.TestCase):
    def test_ffprobe_reads_duration_and_streams(self) -> None:
        response = SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [
                        {"codec_type": "video", "duration": "99.9"},
                        {"codec_type": "audio", "duration": "99.8"},
                    ],
                    "format": {"duration": "100.0"},
                }
            ),
            stderr="",
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.mp4"
            source.write_bytes(b"input")
            with patch("lineup.media.subprocess.run", return_value=response):
                info = probe_media(source)

        self.assertEqual(info.duration_seconds, 100.0)
        self.assertEqual(info.audio_stream_count, 1)
        self.assertEqual(info.video_stream_count, 1)


if __name__ == "__main__":
    unittest.main()
