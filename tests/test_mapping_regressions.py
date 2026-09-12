from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mapping.consensus import resolve_frames
from mapping.csv_export import export_results
from mapping.frame_selector import select_top_frames
from mapping.layout import analyze_layout
from mapping.number_reader import recover_frame_numbers
from mapping.schema import FrameAnalysis, Panel, PlayerObservation
from mapping.text import clean_player_name, is_player_name


def box(x: int, y: int, width: int = 120, height: int = 20) -> list[list[int]]:
    return [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]


class MetadataRegressionTests(unittest.TestCase):
    def test_club_title_and_broadcast_mark_are_not_players(self) -> None:
        self.assertFalse(is_player_name("CHELSEA FC"))
        self.assertFalse(is_player_name("LOSC LILLE"))
        self.assertFalse(is_player_name("Trực tiếp"))

    def test_role_tag_is_removed_from_canonical_name(self) -> None:
        self.assertEqual(clean_player_name("G. DONNARUMMA (GK)"), "G. DONNARUMMA")
        self.assertEqual(clean_player_name("VAN DIJK [C]"), "VAN DIJK")

    def test_numberless_formation_is_only_a_weak_scout_candidate(self) -> None:
        names = ["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT"]
        records = []
        for index, name in enumerate(names):
            records.append((name, box(430 + index % 3 * 180, 170 + index // 3 * 150)))
        result = analyze_layout(
            [text for text, _ in records], [.99] * len(records),
            [polygon for _, polygon in records], (720, 1280, 3), 1.0, 100.0,
        )
        self.assertIn("weak_pitch", {panel.role for panel in result.panels})
        self.assertFalse(result.starter_observations)

    def test_fuzzy_substitutes_heading_excludes_the_bench_column(self) -> None:
        records = [("SUBSTITUTFS", box(20, 70, 170))]
        records += [(f"{20 + index} BENCH {index}", box(20, 105 + index * 28, 210))
                    for index in range(11)]
        names = ["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT"]
        for index, name in enumerate(names):
            x, y = 500 + index % 3 * 170, 170 + index // 3 * 150
            records.extend([(str(index + 1), box(x + 35, y - 42, 24)), (name, box(x, y))])
        result = analyze_layout(
            [text for text, _ in records], [.99] * len(records),
            [polygon for _, polygon in records], (720, 1280, 3), 1.0, 100.0,
        )
        self.assertIn("substitutes", {exclusion.role for exclusion in result.exclusions})
        self.assertFalse(any("BENCH" in player.name for player in result.starter_observations))

    def test_commentator_name_is_removed_with_its_lower_third(self) -> None:
        records = [("COMMENTATORS", box(20, 620, 180)),
                   ("JOHN BROADCAST", box(220, 620, 180))]
        result = analyze_layout(
            [text for text, _ in records], [.99] * len(records),
            [polygon for _, polygon in records], (720, 1280, 3), 1.0, 100.0,
        )
        self.assertIn("commentators", {exclusion.role for exclusion in result.exclusions})
        self.assertFalse(result.starter_observations)


class NumberRegressionTests(unittest.TestCase):
    def test_dominant_local_number_survives_single_noisy_variant(self) -> None:
        observation = PlayerObservation(
            1.0, "starter_pitch", "PAVLOVIC", .99, (.4, .5, .5, .54),
        )
        analysis = FrameAnalysis(1.0, [], [observation], 100.0, 1.0, [], {})
        results = [
            (["31"], [.95], []), (["31"], [.90], []),
            (["8"], [.60], []), ([], [], []), ([], [], []), ([], [], []),
        ]
        with patch("mapping.number_reader.run_ocr_batch", return_value=results):
            recover_frame_numbers(np.zeros((720, 1280, 3), dtype=np.uint8), analysis, object())
        self.assertEqual(observation.jersey_number, 31)
        self.assertEqual(observation.number_source, "local_preprocessed_ocr")

    def test_letter_one_is_rejected_when_crop_still_contains_name(self) -> None:
        observation = PlayerObservation(
            1.0, "starter_pitch", "RICCI", .99, (.4, .5, .5, .54),
        )
        analysis = FrameAnalysis(1.0, [], [observation], 100.0, 1.0, [], {})
        result = (["L", "RICCI"], [.95, .99], [])
        with patch("mapping.number_reader.run_ocr_batch", return_value=[result] * 6):
            recover_frame_numbers(np.zeros((720, 1280, 3), dtype=np.uint8), analysis, object())
        self.assertIsNone(observation.jersey_number)
        self.assertEqual(observation.number_candidates, [])

    def test_duplicate_number_keeps_only_clear_evidence_winner(self) -> None:
        frames = []
        for timestamp in (1.0, 2.0):
            strong = PlayerObservation(
                timestamp, "starter_pitch", "WILLIAMS", .99, (.2, .3, .3, .34),
                jersey_number=17, number_confidence=.98, number_source="spatial_ocr",
                pair_confidence=.95, number_candidates=[17], number_candidate_scores={17: 1.0},
            )
            weak = PlayerObservation(
                timestamp, "starter_pitch", "MORATA", .99, (.6, .3, .7, .34),
                jersey_number=17, number_confidence=.55, number_source="local_preprocessed_ocr",
                pair_confidence=.45, number_candidates=[17], number_candidate_scores={17: .3},
            )
            frames.append(FrameAnalysis(timestamp, [], [strong, weak], 100.0, 1.0, [], {}))
        players, issues = resolve_frames(frames)
        williams = next(player for player in players if player["name"] == "WILLIAMS")
        morata = next(player for player in players if player["name"] == "MORATA")
        self.assertEqual(williams["shirt_number"], 17)
        self.assertIsNone(morata["shirt_number"])
        self.assertIn("duplicate_shirt_numbers", issues)

    def test_centered_edge_goalkeeper_resolves_one_vs_eleven(self) -> None:
        frames = []
        for timestamp in (1.0, 2.0, 3.0):
            goalkeeper = PlayerObservation(
                timestamp, "starter_pitch", "SANCHEZ", .99, (.45, .85, .55, .89),
                number_candidates=[1, 11], number_candidate_scores={1: 1.0, 11: 1.15},
            )
            outfield = PlayerObservation(
                timestamp, "starter_pitch", "PALMER", .99, (.45, .2, .55, .24),
                jersey_number=10, number_confidence=.99, number_source="spatial_ocr",
                pair_confidence=.95, number_candidates=[10], number_candidate_scores={10: 1.0},
            )
            frames.append(FrameAnalysis(timestamp, [], [goalkeeper, outfield], 100.0, 1.0, [], {}))
        players, _ = resolve_frames(frames)
        goalkeeper = next(player for player in players if player["name"] == "SANCHEZ")
        self.assertEqual(goalkeeper["shirt_number"], 1)
        self.assertEqual(goalkeeper["number_status"], "goalkeeper_one_consensus")

    def test_candidate_strength_beats_frequent_weak_noise(self) -> None:
        frames = []
        for timestamp in (1.0, 2.0, 3.0):
            observation = PlayerObservation(
                timestamp, "starter_pitch", "DE WINTER", .99, (.4, .4, .5, .44),
                number_candidates=[5, 8], number_candidate_scores={5: 2.5, 8: 1.0},
            )
            frames.append(FrameAnalysis(timestamp, [], [observation], 100.0, 1.0, [], {}))
        players, _ = resolve_frames(frames)
        self.assertEqual(players[0]["shirt_number"], 5)


class ExportRegressionTests(unittest.TestCase):
    def test_partial_csv_retains_confirmed_pairs(self) -> None:
        confirmed = {
            "slot_index": 1, "shirt_number": 1, "name": "ALISSON", "confirmed": True,
            "name_confidence": .99, "pair_confidence": .95, "evidence_timestamps": [1, 2],
            "number_source": "spatial_ocr",
        }
        extraction = {
            "video_path": "match.mp4",
            "graphics": [{
                "graphic_index": 1, "start_seconds": 0, "end_seconds": 2,
                "best_frame_timestamp": 1, "selected_frame_timestamps": [1, 2],
                "selection_tier": 3, "layout": "pitch", "status": "partial",
                "issues": ["unconfirmed_numbers"], "players": [confirmed],
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = export_results([extraction], directory)
            with paths["partial"].open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual([(row["player_name"], row["shirt_number"]) for row in rows], [("ALISSON", "1")])


class FrameSelectionRegressionTests(unittest.TestCase):
    def test_top_k_uses_quality_instead_of_neighbours_of_one_frame(self) -> None:
        panel = Panel("starter_pitch", (0, 0, 1, 1), [])
        samples = [SimpleNamespace(
            timestamp=float(index),
            analysis=FrameAnalysis(float(index), [panel], [], 100.0, score, [], {}),
        ) for index, score in enumerate((.4, 1.8, .6, 1.6, .5))]
        selected = select_top_frames(samples, 2)
        self.assertEqual([sample.timestamp for sample in selected], [1.0, 3.0])


if __name__ == "__main__":
    unittest.main()
