from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mapping.consensus import resolve_frames
from mapping.csv_export import export_results
from mapping.layout import analyze_layout
from mapping.pipeline import centered_candidates, consolidate_graphics
from mapping.schema import FrameAnalysis, PlayerObservation
from mapping.text import names_compatible, parse_number_crop


NAMES = [
    "ALISSON", "TRENT", "KONATE", "VAN DIJK", "ROBERTSON", "GRAVENBERCH",
    "MAC ALLISTER", "SZOBOSZLAI", "SALAH", "WIRTZ", "EKITIKE",
]


def box(x: int, y: int, width: int = 110, height: int = 20) -> list[list[int]]:
    return [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]


def pitch_records(offset: int = 0) -> list[tuple[str, list[list[int]]]]:
    records = []
    index = 0
    for y, xs in [(150, [700]), (270, [520, 720, 920]), (400, [520, 720, 920]),
                  (530, [520, 720, 920]), (660, [720])]:
        for x in xs:
            records.extend([(str(index + 1), box(x + 35 + offset, y - 42, 24)),
                            (NAMES[index], box(x + offset, y, 130))])
            index += 1
    return records


def infer(records: list[tuple[str, list[list[int]]]], timestamp: float = 1.0) -> FrameAnalysis:
    return analyze_layout(
        [text for text, _ in records], [.99] * len(records), [polygon for _, polygon in records],
        (720, 1280, 3), timestamp, 100.0,
    )


class PanelTests(unittest.TestCase):
    def test_substitutes_and_coach_are_excluded_from_pitch(self) -> None:
        records = [("SUBSTITUTES", box(20, 70, 170))]
        records += [(f"{20 + i} BENCH {i}", box(20, 105 + i * 28, 210)) for i in range(11)]
        records += [("HEAD COACH", box(20, 500, 150)), ("STAFF PERSON", box(20, 535, 170))]
        records += pitch_records()
        result = infer(records)
        self.assertEqual([p.role for p in result.panels if p.role.startswith("starter_")], ["starter_pitch"])
        self.assertEqual([(p.name, p.jersey_number) for p in result.starter_observations],
                         list(zip(NAMES, range(1, 12))))

    def test_matching_list_and_pitch_are_both_starter_evidence(self) -> None:
        records = []
        for i, name in enumerate(NAMES):
            records.extend([(str(i + 1), box(35, 95 + i * 45, 25)),
                            (name, box(75, 95 + i * 45, 150))])
        records += pitch_records()
        roles = {panel.role for panel in infer(records).panels}
        self.assertIn("starter_list", roles)
        self.assertIn("starter_pitch", roles)

    def test_table_only_graphic_is_a_starter_list(self) -> None:
        records = [(f"{i + 1} {name}", box(350, 90 + i * 48, 220))
                   for i, name in enumerate(NAMES)]
        result = infer(records)
        self.assertEqual([panel.role for panel in result.panels], ["starter_list"])
        self.assertEqual([(player.name, player.jersey_number) for player in result.starter_observations],
                         list(zip(NAMES, range(1, 12))))

    def test_substitutes_only_graphic_has_no_starter_panel(self) -> None:
        records = [("SUBSTITUTES", box(340, 70, 170))]
        records += [(f"{20 + i} RESERVE {chr(65 + i)}", box(340, 110 + i * 45, 240))
                    for i in range(11)]
        result = infer(records)
        self.assertFalse(result.starter_observations)
        self.assertIn("no_starter_panel", result.issues)

    def test_disjoint_side_list_is_not_promoted_when_pitch_exists(self) -> None:
        records = []
        for i in range(11):
            records.append((f"RESERVE PERSON {chr(65 + i)}", box(55, 95 + i * 45, 190)))
        records += pitch_records()
        roles = {panel.role for panel in infer(records).panels}
        self.assertIn("auxiliary_list", roles)
        self.assertIn("starter_pitch", roles)


class ConsensusTests(unittest.TestCase):
    def frames(self) -> list[FrameAnalysis]:
        return [infer(pitch_records(), timestamp) for timestamp in (1.0, 1.5, 2.0)]

    def test_two_independent_timestamps_confirm_eleven_pairs(self) -> None:
        players, issues = resolve_frames(self.frames())
        self.assertEqual(issues, [])
        self.assertEqual(len(players), 11)
        self.assertTrue(all(player["confirmed"] for player in players))

    def test_reliable_majority_survives_one_bad_frame(self) -> None:
        frames = self.frames()
        frames[-1].starter_observations[0].jersey_number = 14
        frames[-1].starter_observations[0].number_candidates = [14]
        players, issues = resolve_frames(frames)
        alisson = next(player for player in players if player["name"] == "ALISSON")
        self.assertEqual(alisson["shirt_number"], 1)
        self.assertNotIn("unconfirmed_numbers", issues)

    def test_two_equally_reliable_numbers_remain_unresolved(self) -> None:
        frames = self.frames()[:2]
        frames[-1].starter_observations[0].jersey_number = 14
        frames[-1].starter_observations[0].number_candidates = [14]
        players, issues = resolve_frames(frames)
        alisson = next(player for player in players if player["name"] == "ALISSON")
        self.assertIsNone(alisson["shirt_number"])
        self.assertEqual(alisson["number_status"], "conflict")
        self.assertIn("unconfirmed_numbers", issues)

    def test_short_and_full_name_are_the_same_player(self) -> None:
        self.assertTrue(names_compatible("KONE", "MANU KONE"))
        self.assertTrue(names_compatible("WESLEY", "WESLEY FRANÇA"))

    def test_isolated_number_crop_accepts_common_digit_glyphs(self) -> None:
        self.assertEqual(parse_number_crop("ll"), 11)
        self.assertEqual(parse_number_crop("I7"), 17)


class SelectionAndExportTests(unittest.TestCase):
    def test_candidate_window_is_centered_on_best_frame(self) -> None:
        group = list(range(11, 23, 2))
        self.assertEqual(centered_candidates(group, 2, 3), [13, 15, 17])
        self.assertEqual(centered_candidates([11, 15, 19, 21], 2, 3), [15, 19, 21])
        self.assertEqual(centered_candidates([11, 15, 19, 21], 2, 7), [11, 15, 19, 21])

    def test_resolved_csv_contains_only_complete_graphics(self) -> None:
        player = {
            "slot_index": 1, "shirt_number": 1, "name": "ALISSON", "confirmed": True,
            "name_confidence": .99, "pair_confidence": .95, "evidence_timestamps": [1, 2],
            "number_source": "spatial_ocr",
        }
        base = {
            "video_path": "match.mp4",
            "graphics": [
                {"graphic_index": 1, "start_seconds": 0, "end_seconds": 2,
                 "best_frame_timestamp": 1, "selected_frame_timestamps": [0, 1, 2],
                 "selection_tier": 3, "layout": "pitch", "status": "partial",
                 "issues": ["unconfirmed_numbers"], "players": [player]},
                {"graphic_index": 2, "start_seconds": 3, "end_seconds": 5,
                 "best_frame_timestamp": 4, "selected_frame_timestamps": [3, 4, 5],
                 "selection_tier": 3, "layout": "pitch", "status": "complete",
                 "issues": [], "players": [player]},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = export_results([base], directory)
            with paths["resolved"].open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["graphic_index"], "2")

    def test_list_and_pitch_phases_for_same_team_are_consolidated(self) -> None:
        def player(name: str, number: int | None) -> dict[str, object]:
            return {
                "slot_index": 1, "shirt_number": number, "name": name,
                "name_confirmed": True, "number_confirmed": number is not None,
                "confirmed": number is not None, "name_confidence": .98,
                "pair_confidence": .9, "number_status": "weighted_consensus",
                "number_candidates": [] if number is None else [number],
                "number_source": "spatial_ocr" if number is not None else None,
                "evidence_timestamps": [1.0], "normalized_center": [.5, .5],
                "observations": [],
            }

        def graphic(index: int, layout: str, names: list[str], missing: int | None = None) -> dict[str, object]:
            players = [player(name, None if i == missing else i + 1) for i, name in enumerate(names)]
            return {
                "graphic_index": index, "start_seconds": float(index),
                "end_seconds": float(index) + 1, "best_frame_timestamp": float(index),
                "selected_frame_timestamps": [float(index)], "selection_tier": 3,
                "panel_roles": [f"starter_{layout}"], "layout": layout,
                "status": "partial", "issues": ["unconfirmed_numbers"],
                "players": players, "local_number_evidence": [], "ocr_evidence": [],
            }

        list_phase = graphic(1, "list", NAMES, missing=0)
        pitch_phase = graphic(2, "pitch", ["ALISSON BECKER", *NAMES[1:]])
        merged = consolidate_graphics([list_phase, pitch_phase])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["status"], "complete")
        self.assertEqual(merged[0]["players"][0]["name"], "ALISSON BECKER")
        self.assertEqual(merged[0]["players"][0]["shirt_number"], 1)


if __name__ == "__main__":
    unittest.main()
