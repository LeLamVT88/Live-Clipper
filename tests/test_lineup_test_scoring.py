from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lineup_test.interval_proposer import propose_lineups
from lineup_test.scorer import compute_lineup_score


class LineupScoringTests(unittest.TestCase):
    def test_sponsor_words_and_one_number_do_not_pass_threshold(self) -> None:
        score, details = compute_lineup_score(
            [
                "Expedia",
                "standard chartered",
                "Essential free travel",
                "26",
            ],
            [0.99, 0.99, 0.99, 0.99],
            8,
        )

        self.assertLess(score, 0.45)
        self.assertFalse(details["strong_evidence"])

    def test_formation_is_not_counted_as_a_jersey_number(self) -> None:
        score, details = compute_lineup_score(
            ["FRANCE", "4-3-3", "16 MAIGNAN", "RABIOT KANTE TCHOUAMENI"],
            [0.99, 0.99, 0.99, 0.99],
            9,
        )

        self.assertEqual(details["sample_numbers"], ["16"])
        self.assertTrue(details["strong_evidence"])
        self.assertGreaterEqual(score, 0.45)

    def test_repeated_words_do_not_fake_name_density(self) -> None:
        score, details = compute_lineup_score(
            ["standard chartered standard chartered", "Expedia Expedia"],
            [0.99, 0.99],
            6,
        )

        self.assertEqual(score, 0.0)
        self.assertEqual(details["names"], 0)


class LineupProposalTests(unittest.TestCase):
    def test_single_false_positive_cannot_create_a_proposal(self) -> None:
        result = propose_lineups(
            [10.0, 15.0, 20.0],
            [0.0, 0.8, 0.0],
            [(0.0, 30.0)],
            strong_flags=[False, True, False],
        )

        self.assertEqual(result, [])

    def test_negative_sample_splits_noise_from_real_lineup(self) -> None:
        result = propose_lineups(
            [268.6, 271.84, 275.08, 279.32, 285.33, 290.34, 295.34, 300.35],
            [0.45, 0.2812, 0.45, 0.0, 0.5354, 0.7083, 0.8229, 1.0],
            [(265.36, 278.32), (278.32, 280.32), (280.32, 305.36)],
            strong_flags=[False, False, False, False, True, True, True, True],
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].start_seconds, 280.32)
        self.assertEqual(result[0].end_seconds, 305.36)
        self.assertGreater(result[0].confidence, 0.75)

    def test_adjacent_team_sequences_merge_after_negative_split(self) -> None:
        result = propose_lineups(
            [276.47, 280.61, 283.50, 286.34, 291.72, 299.69, 307.66, 315.63, 323.60],
            [0.45, 1.0, 1.0, 0.0, 0.5687, 0.5667, 0.45, 0.45, 1.0],
            [(268.37, 279.17), (279.17, 284.94), (284.94, 287.74), (287.74, 327.58)],
            strong_flags=[False, True, True, False, False, False, False, False, True],
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].start_seconds, 268.37)
        self.assertEqual(result[0].end_seconds, 327.58)
        self.assertTrue(result[0].metadata["merged_adjacent"])


if __name__ == "__main__":
    unittest.main()
