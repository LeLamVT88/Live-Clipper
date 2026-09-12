from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mapping.config import MappingConfig, PlayerDetectionConfig
from mapping.csv_export import CSV_COLUMNS, write_csv
from mapping.player_detection import detect_player_blobs
from mapping.regions import crop_lineup_region
from mapping.schema import FrameSample, LineupRow, RawPlayer, Rect, SquadPlayer, TeamObservation
from mapping.squad import match_against_squad
from mapping.stability import find_stable_segments, select_representative_frame
from mapping.team import detect_team_boundary, match_team_name


def config(**overrides) -> MappingConfig:
    values = {
        "team_name_bar": Rect(0, 0, 1, .2),
        "lineup_region": Rect(.2, .2, .7, .7),
        "substitutes_region": Rect(0, .2, .2, .7),
        "coach_region": Rect(0, .9, .2, .1),
        "exclude_regions": (Rect(.8, .2, .2, .2),),
    }
    values.update(overrides)
    return MappingConfig(**values)


class StabilityTests(unittest.TestCase):
    def test_find_and_select_middle_of_longest_stable_segment(self) -> None:
        images = [np.full((40, 60, 3), value, np.uint8) for value in (0, 0, 0, 80, 80, 80, 80)]
        frames = [FrameSample(index * .3, image) for index, image in enumerate(images)]
        segments = find_stable_segments(frames, max_mean_diff=1, min_frames=3)
        self.assertEqual([(item.start_index, item.end_index) for item in segments], [(0, 2), (3, 6)])
        representative = select_representative_frame(frames, segments, 0, 3)
        self.assertEqual(representative.timestamp, 1.5)


class TeamTests(unittest.TestCase):
    def test_fuzzy_team_match(self) -> None:
        team, score = match_team_name("BELGIUM.", ["Belgium", "France"], 70)
        self.assertEqual(team, "Belgium")
        self.assertGreaterEqual(score, 70)

    def test_single_bad_frame_does_not_create_boundary(self) -> None:
        observations = [
            TeamObservation(0, "A", "A", 100), TeamObservation(.3, "A", "A", 100),
            TeamObservation(.6, "B", "B", 100),
            TeamObservation(.9, "A", "A", 100), TeamObservation(1.2, "A", "A", 100),
            TeamObservation(1.5, "B", "B", 100), TeamObservation(1.8, "B", "B", 100),
        ]
        boundaries = detect_team_boundary(observations, debounce_frames=2)
        self.assertEqual(len(boundaries), 1)
        self.assertEqual((boundaries[0].from_team, boundaries[0].to_team), ("A", "B"))
        self.assertEqual(boundaries[0].left_seconds, 1.2)
        self.assertEqual(boundaries[0].right_seconds, 1.5)


class RegionAndBlobTests(unittest.TestCase):
    def test_crop_uses_percentage_and_masks_exclusion(self) -> None:
        frame = np.full((100, 100, 3), 255, np.uint8)
        cropped = crop_lineup_region(frame, config())
        self.assertEqual(cropped.shape[:2], (70, 70))
        self.assertTrue(np.all(cropped[:20, 60:] == 0))

    def test_coloured_shirts_become_sorted_player_units(self) -> None:
        image = np.zeros((300, 400, 3), np.uint8)
        cv2.rectangle(image, (70, 40), (110, 90), (0, 0, 220), -1)
        cv2.rectangle(image, (250, 160), (290, 210), (220, 0, 0), -1)
        detection = PlayerDetectionConfig(
            min_area_pct=.01, max_area_pct=.03, morphology_kernel=3,
        )
        units = detect_player_blobs(image, detection)
        self.assertEqual(len(units), 2)
        self.assertLess(units[0].shirt_box[1], units[1].shirt_box[1])
        self.assertGreater(units[0].name_box[1], units[0].number_box[1])


class SquadAndCSVTests(unittest.TestCase):
    def test_name_match_corrects_wrong_ocr_number_and_is_one_to_one(self) -> None:
        raw = [
            RawPlayer(8, "R LUKAKU", .8, .9),
            RawPlayer(7, "K DE BRUYNE", .9, .9),
        ]
        squad = [
            SquadPlayer("Belgium", 10, "R. Lukaku"),
            SquadPlayer("Belgium", 7, "K. De Bruyne"),
        ]
        matched = match_against_squad(raw, "Belgium", squad, threshold=65)
        self.assertEqual(matched, [(10, "R. Lukaku", "high"), (7, "K. De Bruyne", "high")])

    def test_csv_has_required_schema(self) -> None:
        rows = [LineupRow("Belgium", 10, "R. Lukaku", "starting", "clip.mp4", 3.2, "high")]
        with tempfile.TemporaryDirectory() as directory:
            path = write_csv(rows, Path(directory) / "result.csv")
            with path.open(encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                loaded = list(reader)
            self.assertEqual(reader.fieldnames, CSV_COLUMNS)
            self.assertEqual(loaded[0]["role"], "starting")


if __name__ == "__main__":
    unittest.main()

