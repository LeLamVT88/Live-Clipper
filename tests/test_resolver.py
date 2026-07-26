from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ocr import resolver
import ocr.resolver.formation as formation_resolver
import ocr.resolver.local_models as local_ocr_resolver


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
        "frame_index": frame_index,
        "timestamp_seconds": timestamp_seconds,
        "text": text,
        "text_type": text_type,
        "score": 0.99,
        "center_x_norm": x,
        "center_y_norm": y,
    }


class SubstitutePanelMaskTests(unittest.TestCase):
    def test_left_substitutes_panel_is_removed_from_following_frames(self) -> None:
        rows = [
            detection(
                frame_index=1,
                timestamp_seconds=10.0,
                text="SUBSTITUTES",
                text_type="text",
                x=0.15,
                y=0.15,
            ),
            detection(
                frame_index=1,
                timestamp_seconds=10.0,
                text="30 CHEVALIER",
                text_type="text",
                x=0.18,
                y=0.25,
            ),
            detection(
                frame_index=1,
                timestamp_seconds=10.0,
                text="7",
                text_type="shirt_number_candidate",
                x=0.60,
                y=0.30,
            ),
            detection(
                frame_index=1,
                timestamp_seconds=10.0,
                text="KVARATSKHELIA",
                text_type="text",
                x=0.60,
                y=0.37,
            ),
            # Simulate one OCR frame where the substitutes header was missed.
            detection(
                frame_index=2,
                timestamp_seconds=10.5,
                text="89 MARIN",
                text_type="text",
                x=0.18,
                y=0.25,
            ),
            detection(
                frame_index=2,
                timestamp_seconds=10.5,
                text="10",
                text_type="shirt_number_candidate",
                x=0.72,
                y=0.30,
            ),
            detection(
                frame_index=2,
                timestamp_seconds=10.5,
                text="O. DEMBÉLÉ",
                text_type="text",
                x=0.72,
                y=0.37,
            ),
        ]

        filtered = resolver.detections_without_substitute_panel(
            pd.DataFrame(rows)
        )

        self.assertEqual(
            filtered["text"].tolist(),
            ["7", "KVARATSKHELIA", "10", "O. DEMBÉLÉ"],
        )

    def test_right_substitutes_panel_keeps_left_formation(self) -> None:
        rows = [
            detection(
                frame_index=1,
                timestamp_seconds=5.0,
                text="9",
                text_type="shirt_number_candidate",
                x=0.25,
                y=0.30,
            ),
            detection(
                frame_index=1,
                timestamp_seconds=5.0,
                text="STRIKER",
                text_type="text",
                x=0.25,
                y=0.37,
            ),
            detection(
                frame_index=1,
                timestamp_seconds=5.0,
                text="SUBSTITUTES",
                text_type="text",
                x=0.85,
                y=0.15,
            ),
            detection(
                frame_index=1,
                timestamp_seconds=5.0,
                text="18 RESERVE",
                text_type="text",
                x=0.82,
                y=0.25,
            ),
        ]

        filtered = resolver.detections_without_substitute_panel(
            pd.DataFrame(rows)
        )

        self.assertEqual(filtered["text"].tolist(), ["9", "STRIKER"])


class FormationResolutionTests(unittest.TestCase):
    def test_sponsor_text_is_not_treated_as_player_name(self) -> None:
        self.assertFalse(resolver.is_name_like("crypto.Hisense"))
        self.assertFalse(resolver.is_name_like("vivo"))
        self.assertFalse(resolver.is_name_like("TEAM"))
        self.assertFalse(resolver.is_name_like("TEAM FORMATION"))

    def test_interface_title_is_not_appended_to_player_name(self) -> None:
        detections = pd.DataFrame(
            [
                detection(
                    frame_index=1,
                    timestamp_seconds=1.0,
                    text="MINKI",
                    text_type="text",
                    x=0.45,
                    y=0.60,
                ),
                detection(
                    frame_index=1,
                    timestamp_seconds=1.0,
                    text="TEAM",
                    text_type="text",
                    x=0.45,
                    y=0.66,
                ),
                detection(
                    frame_index=1,
                    timestamp_seconds=1.0,
                    text="FORMATION",
                    text_type="text",
                    x=0.45,
                    y=0.72,
                ),
            ]
        )

        player_name, _, _ = resolver.best_full_name(
            "MINKI",
            detections,
            other_formation_labels=set(),
        )

        self.assertEqual(player_name, "MINKI")

    def test_numeric_consensus_outvotes_single_wrong_detector(self) -> None:
        result = local_ocr_resolver.numeric_candidate_consensus(
            [
                (1, 0.99),
                (7, 0.81),
                (7, 0.91),
                (7, 0.95),
            ]
        )

        self.assertIsNotNone(result)
        self.assertEqual(result[0], 7)

    def test_duplicate_shirt_numbers_are_rejected(self) -> None:
        records = [
            {"shirt_number": 1},
            {"shirt_number": 7},
            {"shirt_number": 1},
        ]

        with self.assertRaisesRegex(
            resolver.LineupResolutionError,
            "Duplicate shirt number",
        ):
            resolver.validate_unique_shirt_numbers(
                records,
                lineup_index=1,
            )

    def test_candidate_names_include_low_goalkeeper_label(self) -> None:
        rows = [
            detection(
                frame_index=1,
                timestamp_seconds=10.0,
                text="20",
                text_type="shirt_number_candidate",
                x=0.45,
                y=0.25,
            ),
            detection(
                frame_index=1,
                timestamp_seconds=10.0,
                text="WINGER",
                text_type="text",
                x=0.45,
                y=0.31,
            ),
            detection(
                frame_index=1,
                timestamp_seconds=10.0,
                text="3",
                text_type="shirt_number_candidate",
                x=0.70,
                y=0.62,
            ),
            detection(
                frame_index=1,
                timestamp_seconds=10.0,
                text="DEFENDER",
                text_type="text",
                x=0.70,
                y=0.68,
            ),
            detection(
                frame_index=1,
                timestamp_seconds=10.0,
                text="GOALKEEPER NAME",
                text_type="text",
                x=0.60,
                y=0.94,
            ),
        ]
        frame = pd.DataFrame(rows)
        anchors = resolver.formation_anchor_pairs(frame)

        names = resolver.formation_name_rows(frame, anchors)

        self.assertIn("GOALKEEPER NAME", [str(row["text"]) for row in names])

    def test_candidate_names_include_wide_players_outside_anchor_span(
        self,
    ) -> None:
        rows = [
            detection(
                frame_index=1,
                timestamp_seconds=10.0,
                text="5",
                text_type="shirt_number_candidate",
                x=0.50,
                y=0.40,
            ),
            detection(
                frame_index=1,
                timestamp_seconds=10.0,
                text="CENTRAL PLAYER",
                text_type="text",
                x=0.50,
                y=0.46,
            ),
            detection(
                frame_index=1,
                timestamp_seconds=10.0,
                text="LEFT WINGER",
                text_type="text",
                x=0.05,
                y=0.30,
            ),
            detection(
                frame_index=1,
                timestamp_seconds=10.0,
                text="RIGHT WINGER",
                text_type="text",
                x=0.95,
                y=0.30,
            ),
        ]
        frame = pd.DataFrame(rows)
        anchors = resolver.formation_anchor_pairs(frame)

        names = {
            str(row["text"])
            for row in resolver.formation_name_rows(frame, anchors)
        }

        self.assertIn("LEFT WINGER", names)
        self.assertIn("RIGHT WINGER", names)

    def test_table_rows_accept_split_and_inline_player_boxes(self) -> None:
        rows = [
            detection(
                frame_index=1,
                timestamp_seconds=10.0,
                text="1 FIRST PLAYER",
                text_type="text",
                x=0.16,
                y=0.30,
            ),
            detection(
                frame_index=1,
                timestamp_seconds=10.0,
                text="SECOND PLAYER",
                text_type="text",
                x=0.19,
                y=0.36,
            ),
            detection(
                frame_index=1,
                timestamp_seconds=10.0,
                text="3 THIRD PLAYER",
                text_type="text",
                x=0.16,
                y=0.42,
            ),
        ]
        for row, x1 in zip(rows, (90, 125, 90), strict=True):
            row["x1"] = x1
            row["frame_width"] = 1000
        observations = [
            resolver.PairObservation(
                shirt_number=2,
                player_name="SECOND PLAYER",
                number_score=0.99,
                name_score=0.99,
                frame_index=1,
                timestamp_seconds=10.0,
                center_x=0.10,
                center_y=0.36,
                label_x1=0.125,
                relation="right",
            ),
            resolver.PairObservation(
                shirt_number=4,
                player_name="ANOTHER PLAYER",
                number_score=0.99,
                name_score=0.99,
                frame_index=1,
                timestamp_seconds=10.0,
                center_x=0.10,
                center_y=0.48,
                label_x1=0.125,
                relation="right",
            ),
        ]

        table_rows = resolver.table_rows_for_frame(
            pd.DataFrame(rows),
            observations,
            expected_players=3,
        )

        self.assertEqual(
            [str(row["text"]) for row in table_rows],
            ["1 FIRST PLAYER", "SECOND PLAYER", "3 THIRD PLAYER"],
        )

    def test_table_consensus_uses_only_high_confidence_singleton_fill(
        self,
    ) -> None:
        def pair(
            number: int,
            name: str,
            frame_index: int,
            score: float,
        ) -> resolver.PairObservation:
            return resolver.PairObservation(
                shirt_number=number,
                player_name=name,
                number_score=score,
                name_score=score,
                frame_index=frame_index,
                timestamp_seconds=float(frame_index),
                center_x=0.10,
                center_y=0.20 + number * 0.05,
                label_x1=0.125,
                relation="right",
            )

        selected = resolver.select_table_pairs(
            [
                pair(1, "FIRST PLAYER", 1, 0.99),
                pair(1, "FIRST PLAYER", 2, 0.99),
                pair(2, "SECOND PLAYER", 1, 0.98),
                pair(3, "THIRD PLAYER", 1, 0.70),
            ],
            expected_players=3,
        )

        self.assertEqual(
            {group[0].shirt_number for group in selected},
            {1, 2},
        )

    def test_number_gap_is_learned_from_current_layout(self) -> None:
        compact = pd.DataFrame(
            [
                detection(
                    frame_index=1,
                    timestamp_seconds=1.0,
                    text="7",
                    text_type="shirt_number_candidate",
                    x=0.4,
                    y=0.20,
                ),
                detection(
                    frame_index=1,
                    timestamp_seconds=1.0,
                    text="PLAYER A",
                    text_type="text",
                    x=0.4,
                    y=0.26,
                ),
            ]
        )
        tall = pd.DataFrame(
            [
                detection(
                    frame_index=1,
                    timestamp_seconds=1.0,
                    text="9",
                    text_type="shirt_number_candidate",
                    x=0.6,
                    y=0.20,
                ),
                detection(
                    frame_index=1,
                    timestamp_seconds=1.0,
                    text="PLAYER B",
                    text_type="text",
                    x=0.6,
                    y=0.32,
                ),
            ]
        )

        compact_gap = resolver.formation_number_gap(
            resolver.formation_anchor_pairs(compact)
        )
        tall_gap = resolver.formation_number_gap(
            resolver.formation_anchor_pairs(tall)
        )

        self.assertAlmostEqual(compact_gap, 0.06)
        self.assertAlmostEqual(tall_gap, 0.12)

    def test_failed_partial_event_does_not_increment_lineup_index(self) -> None:
        segment = pd.DataFrame(
            [
                {
                    **detection(
                        frame_index=1,
                        timestamp_seconds=1.0,
                        text="PLAYER",
                        text_type="text",
                        x=0.5,
                        y=0.5,
                    ),
                    "segment_end_seconds": 3.0,
                }
            ]
        )
        events = [
            resolver.FormationEvent([1], [1.0], [[]]),
            resolver.FormationEvent([2], [2.0], [[]]),
        ]

        def fake_resolve_event(*args: object, **kwargs: object) -> list[dict[str, int]]:
            lineup_index = int(kwargs["lineup_index"])
            if fake_resolve_event.calls == 0:
                fake_resolve_event.calls += 1
                raise resolver.LineupResolutionError("partial animation")
            return [{"lineup_index": lineup_index}]

        fake_resolve_event.calls = 0
        with (
            patch.object(
                formation_resolver,
                "detect_formation_events",
                return_value=events,
            ),
            patch.object(
                formation_resolver,
                "resolve_event",
                side_effect=fake_resolve_event,
            ),
        ):
            _, _, records, _ = resolver.attempt_formation_resolution(
                segment,
                expected_players=11,
                min_number_count=8,
                max_gap_seconds=20.0,
                signature_threshold=0.45,
            )

        self.assertEqual(records, [{"lineup_index": 1}])


if __name__ == "__main__":
    unittest.main()
