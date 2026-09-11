from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from line_up.config import PipelineConfig
from line_up.scorer import compute_lineup_score
from line_up.interval_proposer import propose_lineups
from line_up.multimodal import compute_combined_multimodal_score
from line_up.pipeline import get_recovery_timestamps
from line_up.schema import FrameSampleResult, LineupInterval


class IntegrationTests(unittest.TestCase):
    def score(self, texts):
        return compute_lineup_score(texts, [0.99] * len(texts), len(texts))

    def test_lineup_title_with_broadcast_clock(self):
        score, details = self.score(['LINE-UPS', '01:58', '16 MAIGNAN', 'RABIOT KANTE TCHOUAMENI'])
        self.assertGreaterEqual(score, .45)
        self.assertTrue(details['strong_evidence'])

    def test_clock_without_lineup_remains_suppressed(self):
        score, _ = self.score(['我3:42', 'MAIGNAN RABIOT KANTE TCHOUAMENI', '16 8 7'])
        self.assertEqual(score, 0)

    def test_standings_and_multiview_are_content_barriers(self):
        for texts in [
            ['PLD', 'PTS', 'MAINZ AUGSBURG HAMBURG', '28 28 28 -17'],
            ['beIN SPORTS 1', 'beIN SPORTS 4', 'SUBSTITUTES', '11 SALAH 9 KANE'],
            ['VAR', 'SMITH JONES TAYLOR'],
        ]:
            with self.subTest(texts=texts):
                score, details = self.score(texts)
                self.assertEqual(score, 0)
                self.assertTrue(details['content_barrier'])

    def test_motion_cannot_veto_strong_roster(self):
        self.assertEqual(compute_combined_multimodal_score(.8, .2, 1., strong_evidence=True), .8)

    def test_static_geometry_cannot_rescue_semantic_negative(self):
        self.assertEqual(compute_combined_multimodal_score(0., 1., 1., strong_evidence=False), 0.)

    def test_missing_pair_preserves_ocr(self):
        self.assertEqual(compute_combined_multimodal_score(.4, None, 0., strong_evidence=False), .4)

    def test_explicit_barrier_prevents_team_merge(self):
        times = [5., 10., 20., 30., 35.]
        scores = [.9, .9, 0., .9, .9]
        scenes = [(0., 18.), (18., 22.), (22., 40.)]
        flags = [True, True, False, True, True]
        ordinary = propose_lineups(times, scores, scenes, strong_flags=flags)
        blocked = propose_lineups(times, scores, scenes, strong_flags=flags,
                                  content_barriers=[False, False, True, False, False])
        self.assertEqual(len(ordinary), 1)
        self.assertEqual(len(blocked), 2)
        self.assertLess(blocked[0].end_seconds, 20.)
        self.assertGreater(blocked[1].start_seconds, 20.)

    def test_recovery_verifies_uncovered_anchor_only(self):
        f = FrameSampleResult(20., 1., 20, [], [], False, True, strong_evidence=True)
        config = PipelineConfig()
        self.assertEqual(get_recovery_timestamps([f], [], [(0., 40.)], config), [18.5, 21.5])
        self.assertEqual(get_recovery_timestamps([f], [LineupInterval(10., 30., 1.)], [(0., 40.)], config), [])
        f.strong_evidence = False
        self.assertEqual(get_recovery_timestamps([f], [], [(0., 40.)], config), [])


if __name__ == '__main__':
    unittest.main()
