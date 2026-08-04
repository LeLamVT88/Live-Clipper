from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import run_full_pipeline as pipeline

sys.path.insert(0, str(PROJECT_ROOT / "src" / "lineup"))
from extract_frames import select_videos


class FullPipelineTests(unittest.TestCase):
    @patch.object(pipeline, "validate_inputs", return_value=None)
    @patch.object(
        pipeline,
        "selected_videos",
        return_value=["league/match.mp4"],
    )
    @patch.object(pipeline, "segments_exist", return_value=True)
    @patch.object(pipeline, "run_stage", side_effect=[0, 0, 0, 0, 0])
    def test_runs_all_stages_in_order(
        self, run_stage, _segments, _videos, _validate
    ) -> None:
        result = pipeline.main(["--video", "match.mp4", "--device", "cpu"])

        self.assertEqual(result, 0)
        self.assertEqual(run_stage.call_count, 5)
        paths = pipeline.RunPaths.for_video("league/match.mp4")
        first_command = run_stage.call_args_list[0].args[2]
        second_command = run_stage.call_args_list[1].args[2]
        aggregate_command = run_stage.call_args_list[2].args[2]
        export_command = run_stage.call_args_list[3].args[2]
        ocr_command = run_stage.call_args_list[4].args[2]
        self.assertIn("league/match.mp4", first_command)
        self.assertIn(str(paths.metadata_csv), first_command)
        self.assertIn(str(paths.inference_csv), second_command)
        self.assertEqual(second_command[-1], "cpu")
        self.assertEqual(aggregate_command[-2:], ["--expected-segments-per-video", "2"])
        self.assertIn(str(paths.segments_csv), aggregate_command)
        self.assertIn("--overwrite", export_command)
        self.assertIn(str(paths.clips), export_command)
        self.assertEqual(ocr_command[0], str(pipeline.OCR_PYTHON))
        self.assertIn(str(paths.ocr / "resolved_lineups.csv"), ocr_command)

    @patch.object(pipeline, "validate_inputs", return_value=None)
    @patch.object(
        pipeline,
        "selected_videos",
        return_value=["league/match.mp4"],
    )
    @patch.object(pipeline, "segments_exist", return_value=True)
    @patch.object(pipeline, "run_stage", side_effect=[0, 7])
    def test_stops_after_a_failed_stage(
        self, run_stage, _segments, _videos, _validate
    ) -> None:
        result = pipeline.main(["--video", "match.mp4"])

        self.assertEqual(result, 7)
        self.assertEqual(run_stage.call_count, 2)

    @patch.object(pipeline, "validate_inputs", return_value=None)
    @patch.object(
        pipeline,
        "selected_videos",
        return_value=["league/match.mp4"],
    )
    @patch.object(pipeline, "segments_exist", return_value=False)
    @patch.object(pipeline, "run_stage", side_effect=[0, 0, 0])
    def test_skips_ocr_when_no_segments_exist(
        self, run_stage, _segments, _videos, _validate
    ) -> None:
        result = pipeline.main(["--video", "match.mp4"])

        self.assertEqual(result, 0)
        self.assertEqual(run_stage.call_count, 3)

    @patch.object(pipeline, "validate_inputs", return_value=None)
    @patch.object(
        pipeline,
        "selected_videos",
        return_value=["league/match.mp4"],
    )
    @patch.object(pipeline, "segments_exist", return_value=True)
    @patch.object(pipeline, "run_stage", side_effect=[0, 0, 0, 0, 2])
    def test_propagates_partial_ocr_status(
        self, run_stage, _segments, _videos, _validate
    ) -> None:
        result = pipeline.main(["--video", "match.mp4"])

        self.assertEqual(result, 2)

    def test_finds_a_video_inside_a_nested_data_test_directory(self) -> None:
        with TemporaryDirectory() as temporary:
            input_dir = Path(temporary)
            video = input_dir / "worldcup" / "worldcup_match_01.mp4"
            video.parent.mkdir()
            video.touch()

            selected = select_videos(input_dir, [video.name])

            self.assertEqual(selected, [video])


if __name__ == "__main__":
    unittest.main()
