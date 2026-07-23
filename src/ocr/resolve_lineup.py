"""Resolve player names and shirt numbers from multi-frame OCR detections."""

from __future__ import annotations

import argparse
import math
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = (
    PROJECT_ROOT / "outputs" / "predictions" / "ocr_raw_detections.csv"
)
DEFAULT_OUTPUT_CSV = (
    PROJECT_ROOT / "outputs" / "predictions" / "resolved_lineups.csv"
)
REQUIRED_COLUMNS = {
    "video",
    "segment_index",
    "segment_start_seconds",
    "segment_end_seconds",
    "frame_index",
    "timestamp_seconds",
    "text",
    "text_type",
    "score",
    "center_x_norm",
    "center_y_norm",
}
OUTPUT_COLUMNS = [
    "video",
    "segment_index",
    "lineup_index",
    "resolution_method",
    "formation_timestamp_seconds",
    "slot_index",
    "row_index",
    "shirt_number",
    "formation_label",
    "player_name",
    "number_confidence",
    "label_confidence",
    "name_confidence",
    "pair_confidence",
    "number_evidence_frames",
    "label_evidence_frames",
    "full_name_evidence_frames",
    "slot_center_x_norm",
    "slot_center_y_norm",
]
DIAGNOSTIC_COLUMNS = [
    "video",
    "segment_index",
    "status",
    "resolution_method",
    "resolved_players",
    "message",
]
IGNORED_TEXT = {
    "bre",
    "defence",
    "expedia",
    "football club",
    "forwards",
    "goalkeeper",
    "hollywood",
    "lfc",
    "midfield",
    "premier league",
    "standard",
    "standard chartered",
    "substitutes",
}
TABLE_IGNORED_TEXT = IGNORED_TEXT | {
    "captain",
    "coach",
    "df",
    "elite",
    "fw",
    "gk",
    "head coach",
    "mf",
    "team lineup",
}


class LineupResolutionError(Exception):
    """Raised when raw OCR detections cannot be resolved into a lineup."""


@dataclass
class NumberCluster:
    observations: list[pd.Series] = field(default_factory=list)
    label_observations: list[tuple[str, float, int]] = field(default_factory=list)

    @property
    def center_x(self) -> float:
        return float(np.median([float(row["center_x_norm"]) for row in self.observations]))

    @property
    def center_y(self) -> float:
        return float(np.median([float(row["center_y_norm"]) for row in self.observations]))

    @property
    def frame_count(self) -> int:
        return len({int(row["frame_index"]) for row in self.observations})


@dataclass
class FormationEvent:
    snapshot_frames: list[int]
    snapshot_timestamps: list[float]
    signatures: list[list[pd.Series]]

    @property
    def start_seconds(self) -> float:
        return min(self.snapshot_timestamps)

    @property
    def end_seconds(self) -> float:
        return max(self.snapshot_timestamps)


@dataclass(frozen=True)
class PairObservation:
    shirt_number: int
    player_name: str
    number_score: float
    name_score: float
    frame_index: int
    timestamp_seconds: float
    center_x: float
    center_y: float
    label_x1: float
    relation: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate multi-frame OCR, detect formation snapshots, and map "
            "shirt numbers to player names without a roster."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--diagnostics-csv",
        type=Path,
        help=(
            "Per-segment success/failure report. Defaults to "
            "<output stem>_diagnostics.csv."
        ),
    )
    parser.add_argument(
        "--players-per-lineup",
        type=int,
        default=11,
        help="Expected player slots in one formation (default: 11).",
    )
    parser.add_argument(
        "--min-number-count",
        type=int,
        default=8,
        help="Minimum shirt-number detections in a formation snapshot.",
    )
    parser.add_argument(
        "--same-lineup-gap-seconds",
        type=float,
        default=20.0,
        help="Maximum gap between snapshots of the same lineup.",
    )
    parser.add_argument(
        "--signature-threshold",
        type=float,
        default=0.45,
        help="Minimum spatial number-signature similarity for the same lineup.",
    )
    parser.add_argument(
        "--disable-local-ocr",
        "--disable-local-table-ocr",
        dest="disable_local_ocr",
        action="store_true",
        help=(
            "Do not run the small second OCR pass on table cells or formation "
            "shirt numbers."
        ),
    )
    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def similarity(left: object, right: object) -> float:
    left_normalized = normalize_text(left)
    right_normalized = normalize_text(right)
    if not left_normalized or not right_normalized:
        return 0.0
    return SequenceMatcher(None, left_normalized, right_normalized).ratio()


def is_name_like(text: object) -> bool:
    normalized = normalize_text(text)
    if not normalized or normalized in IGNORED_TEXT:
        return False
    letters = sum(character.isalpha() for character in normalized)
    if letters < 2 or len(normalized) > 40:
        return False
    return not normalized.isdigit()


def is_table_name_like(text: object) -> bool:
    normalized = normalize_text(text)
    if normalized in TABLE_IGNORED_TEXT:
        return False
    return is_name_like(text)


def parse_inline_player(text: object) -> tuple[int, str] | None:
    match = re.match(r"^\s*(\d{1,2})\s+(.+?)\s*$", str(text))
    if not match:
        return None
    shirt_number = int(match.group(1))
    player_name = match.group(2).strip()
    if not 1 <= shirt_number <= 99 or not is_table_name_like(player_name):
        return None
    return shirt_number, player_name


def detections_before_substitutes(segment: pd.DataFrame) -> pd.DataFrame:
    substitute_rows = segment[
        segment["text"].map(normalize_text).str.contains(
            r"\bsubstitutes?\b",
            regex=True,
            na=False,
        )
    ]
    if substitute_rows.empty:
        return segment
    cutoff = float(substitute_rows["timestamp_seconds"].min())
    return segment[segment["timestamp_seconds"] < cutoff]


def detections_without_substitute_panel(segment: pd.DataFrame) -> pd.DataFrame:
    keep = pd.Series(True, index=segment.index)
    for frame_index, frame in segment.groupby("frame_index", sort=False):
        headers = frame[
            frame["text"].map(normalize_text).str.contains(
                r"\bsubstitutes?\b",
                regex=True,
                na=False,
            )
        ]
        if headers.empty:
            continue
        header_x = float(headers["center_x_norm"].median())
        frame_mask = segment["frame_index"] == frame_index
        if header_x < 0.5:
            keep &= ~(
                frame_mask
                & (segment["center_x_norm"] < header_x + 0.15)
            )
        else:
            keep &= ~(
                frame_mask
                & (segment["center_x_norm"] > header_x - 0.15)
            )
    return segment[keep]


def table_pair_observations(segment: pd.DataFrame) -> list[PairObservation]:
    observations: list[PairObservation] = []
    relevant = detections_before_substitutes(segment)
    has_boxes = {
        "x1",
        "frame_width",
    }.issubset(relevant.columns)

    for frame_index, frame in relevant.groupby("frame_index", sort=True):
        timestamp = float(frame["timestamp_seconds"].iloc[0])

        for _, row in frame.iterrows():
            inline = parse_inline_player(row["text"])
            if inline is None:
                continue
            shirt_number, player_name = inline
            label_x1 = (
                float(row["x1"]) / float(row["frame_width"])
                if has_boxes and float(row["frame_width"]) > 0
                else float(row["center_x_norm"])
            )
            observations.append(
                PairObservation(
                    shirt_number=shirt_number,
                    player_name=player_name,
                    number_score=float(row["score"]),
                    name_score=float(row["score"]),
                    frame_index=int(frame_index),
                    timestamp_seconds=timestamp,
                    center_x=float(row["center_x_norm"]),
                    center_y=float(row["center_y_norm"]),
                    label_x1=label_x1,
                    relation="inline",
                )
            )

        numbers = shirt_number_rows(frame)
        names = frame[
            (frame["text_type"] != "shirt_number_candidate")
            & frame["text"].map(is_table_name_like)
            & ~frame["text"].map(lambda value: parse_inline_player(value) is not None)
        ]
        for _, name_row in names.iterrows():
            candidates: list[tuple[float, pd.Series]] = []
            for _, number_row in numbers.iterrows():
                dx = (
                    float(name_row["center_x_norm"])
                    - float(number_row["center_x_norm"])
                )
                dy = abs(
                    float(name_row["center_y_norm"])
                    - float(number_row["center_y_norm"])
                )
                if 0.015 <= dx <= 0.25 and dy <= 0.022:
                    candidates.append((5 * dy + dx, number_row))
            if not candidates:
                continue

            number_row = min(candidates, key=lambda item: item[0])[1]
            label_x1 = (
                float(name_row["x1"]) / float(name_row["frame_width"])
                if has_boxes and float(name_row["frame_width"]) > 0
                else float(name_row["center_x_norm"])
            )
            observations.append(
                PairObservation(
                    shirt_number=int(str(number_row["text"])),
                    player_name=str(name_row["text"]),
                    number_score=float(number_row["score"]),
                    name_score=float(name_row["score"]),
                    frame_index=int(frame_index),
                    timestamp_seconds=timestamp,
                    center_x=float(number_row["center_x_norm"]),
                    center_y=float(name_row["center_y_norm"]),
                    label_x1=label_x1,
                    relation="right",
                )
            )

    return observations


def group_pair_observations(
    observations: list[PairObservation],
) -> list[list[PairObservation]]:
    groups: list[list[PairObservation]] = []
    for observation in observations:
        for group in groups:
            representative = group[0]
            if (
                observation.shirt_number == representative.shirt_number
                and similarity(
                    observation.player_name,
                    representative.player_name,
                )
                >= 0.84
            ):
                group.append(observation)
                break
        else:
            groups.append([observation])
    return groups


def select_table_pairs(
    observations: list[PairObservation],
    expected_players: int,
) -> list[list[PairObservation]]:
    ranked = sorted(
        group_pair_observations(observations),
        key=lambda group: (
            len({observation.frame_index for observation in group}),
            sum(
                math.sqrt(observation.number_score * observation.name_score)
                for observation in group
            ),
        ),
        reverse=True,
    )

    selected: list[list[PairObservation]] = []
    used_numbers: set[int] = set()
    used_names: list[str] = []
    for group in ranked:
        representative = group[0]
        normalized_name = normalize_text(representative.player_name)
        evidence_frames = {
            observation.frame_index for observation in group
        }
        if len(evidence_frames) < 2:
            continue
        if representative.shirt_number in used_numbers:
            continue
        if any(
            similarity(normalized_name, existing) >= 0.84
            for existing in used_names
        ):
            continue
        selected.append(group)
        used_numbers.add(representative.shirt_number)
        used_names.append(normalized_name)
        if len(selected) == expected_players:
            break
    return selected


def resolve_table_layout(
    segment: pd.DataFrame,
    expected_players: int,
    lineup_index: int = 1,
) -> tuple[list[dict[str, object]], int]:
    selected = select_table_pairs(
        table_pair_observations(segment),
        expected_players=expected_players,
    )
    if len(selected) < expected_players:
        return [], len(selected)

    selected.sort(
        key=lambda group: float(
            np.median([observation.center_y for observation in group])
        )
    )
    records: list[dict[str, object]] = []
    for slot_index, group in enumerate(selected, start=1):
        name_observations = [
            (
                observation.player_name,
                observation.name_score,
                observation.frame_index,
            )
            for observation in group
        ]
        player_name, name_confidence, name_evidence = consensus_text(
            name_observations
        )
        number_confidence = float(
            np.mean([observation.number_score for observation in group])
        )
        pair_confidence = float(
            np.mean(
                [
                    math.sqrt(
                        observation.number_score * observation.name_score
                    )
                    for observation in group
                ]
            )
        )
        records.append(
            {
                "video": str(segment["video"].iloc[0]),
                "segment_index": int(segment["segment_index"].iloc[0]),
                "lineup_index": lineup_index,
                "resolution_method": "table",
                "formation_timestamp_seconds": min(
                    observation.timestamp_seconds for observation in group
                ),
                "slot_index": slot_index,
                "row_index": slot_index,
                "shirt_number": group[0].shirt_number,
                "formation_label": player_name,
                "player_name": player_name,
                "number_confidence": round(number_confidence, 6),
                "label_confidence": round(name_confidence, 6),
                "name_confidence": round(name_confidence, 6),
                "pair_confidence": round(pair_confidence, 6),
                "number_evidence_frames": len(
                    {observation.frame_index for observation in group}
                ),
                "label_evidence_frames": name_evidence,
                "full_name_evidence_frames": name_evidence,
                "slot_center_x_norm": round(
                    float(
                        np.median(
                            [observation.center_x for observation in group]
                        )
                    ),
                    6,
                ),
                "slot_center_y_norm": round(
                    float(
                        np.median(
                            [observation.center_y for observation in group]
                        )
                    ),
                    6,
                ),
            }
        )
    return records, len(selected)


def table_rows_for_frame(
    frame: pd.DataFrame,
    observations: list[PairObservation],
    expected_players: int,
) -> list[pd.Series]:
    if not observations or not {
        "x1",
        "frame_width",
    }.issubset(frame.columns):
        return []

    anchor_x1 = float(
        np.median([observation.label_x1 for observation in observations])
    )
    candidates: list[pd.Series] = []
    for _, row in frame.iterrows():
        if (
            row["text_type"] == "shirt_number_candidate"
            or not is_table_name_like(row["text"])
            or parse_inline_player(row["text"]) is not None
        ):
            continue
        frame_width = float(row["frame_width"])
        if frame_width <= 0:
            continue
        x1_normalized = float(row["x1"]) / frame_width
        if abs(x1_normalized - anchor_x1) <= 0.025:
            candidates.append(row)

    candidates.sort(key=lambda row: float(row["center_y_norm"]))
    deduplicated: list[pd.Series] = []
    for row in candidates:
        if (
            not deduplicated
            or float(row["center_y_norm"])
            - float(deduplicated[-1]["center_y_norm"])
            > 0.02
        ):
            deduplicated.append(row)
        elif float(row["score"]) > float(deduplicated[-1]["score"]):
            deduplicated[-1] = row

    if len(deduplicated) < expected_players:
        return []
    best_rows: list[pd.Series] = []
    best_regularity = math.inf
    for start in range(len(deduplicated) - expected_players + 1):
        rows = deduplicated[start : start + expected_players]
        y_values = [float(row["center_y_norm"]) for row in rows]
        gaps = np.diff(y_values)
        median_gap = float(np.median(gaps))
        if not 0.03 <= median_gap <= 0.085:
            continue
        regularity = float(np.mean(np.abs(gaps - median_gap)))
        if regularity < best_regularity:
            best_regularity = regularity
            best_rows = rows
    return best_rows


def create_local_table_ocr():
    try:
        from .run_lineup_ocr import create_ocr
    except ImportError:
        from run_lineup_ocr import create_ocr

    return create_ocr(
        PROJECT_ROOT / ".cache" / "paddlex",
        recognition_batch_size=8,
    )


def create_local_number_recognizer():
    try:
        from paddleocr import TextRecognition
    except ModuleNotFoundError as exc:
        raise LineupResolutionError(
            "PaddleOCR text recognition is not installed."
        ) from exc
    return TextRecognition(
        model_name="PP-OCRv6_small_rec",
        device="cpu",
    )


def local_result_data(result: object) -> dict[str, object]:
    payload = getattr(result, "json", None)
    if callable(payload):
        payload = payload()
    if not isinstance(payload, dict):
        raise LineupResolutionError(
            "Local PaddleOCR returned an unsupported result format."
        )
    data = payload.get("res", payload)
    if not isinstance(data, dict):
        raise LineupResolutionError(
            "Local PaddleOCR result does not contain a result object."
        )
    return data


def refine_table_numbers(
    segment: pd.DataFrame,
    ocr: object,
    expected_players: int,
) -> tuple[pd.DataFrame, int]:
    if "frame_path" not in segment.columns:
        return segment, 0

    observations = [
        observation
        for observation in table_pair_observations(segment)
        if observation.relation == "right"
    ]
    by_frame: defaultdict[int, list[PairObservation]] = defaultdict(list)
    for observation in observations:
        by_frame[observation.frame_index].append(observation)
    usable_frames = [
        (frame_index, frame_observations)
        for frame_index, frame_observations in by_frame.items()
        if len(frame_observations) >= 4
    ]
    if not usable_frames:
        return segment, 0

    earliest = min(usable_frames, key=lambda item: item[0])
    ranked = sorted(
        usable_frames,
        key=lambda item: (len(item[1]), -item[0]),
        reverse=True,
    )
    chosen: list[tuple[int, list[PairObservation]]] = [earliest]
    for candidate in ranked:
        if candidate[0] != earliest[0]:
            chosen.append(candidate)
        if len(chosen) == 3:
            break

    added_rows: list[dict[str, object]] = []
    for frame_index, frame_observations in chosen:
        frame = segment[segment["frame_index"] == frame_index]
        rows = table_rows_for_frame(
            frame,
            frame_observations,
            expected_players=expected_players,
        )
        if len(rows) != expected_players:
            continue

        frame_path = resolve_project_path(Path(str(frame["frame_path"].iloc[0])))
        if not frame_path.is_file():
            continue
        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise LineupResolutionError(
                "OpenCV is required for local table OCR."
            ) from exc

        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        number_center_x = float(
            np.median(
                [observation.center_x for observation in frame_observations]
            )
        )
        y_values = [float(row["center_y_norm"]) for row in rows]
        row_gap = float(np.median(np.diff(y_values)))
        x1 = max(0, int(round((number_center_x - 0.035) * width)))
        x2 = min(width, int(round((number_center_x + 0.035) * width)))
        half_height = max(18, int(round(row_gap * height * 0.43)))

        crops: list[np.ndarray] = []
        for row in rows:
            center_y = int(round(float(row["center_y_norm"]) * height))
            crop = image[
                max(0, center_y - half_height) : min(
                    height,
                    center_y + half_height,
                ),
                x1:x2,
            ]
            crops.append(
                cv2.resize(
                    crop,
                    None,
                    fx=3,
                    fy=3,
                    interpolation=cv2.INTER_CUBIC,
                )
            )

        results = list(ocr.predict(crops))
        if len(results) != len(rows):
            continue
        for name_row, result in zip(rows, results, strict=True):
            data = local_result_data(result)
            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])
            numeric = [
                (int(str(text).strip()), float(score))
                for text, score in zip(texts, scores, strict=False)
                if str(text).strip().isdigit()
                and 1 <= int(str(text).strip()) <= 99
                and float(score) >= 0.60
            ]
            if not numeric:
                continue
            shirt_number, score = max(numeric, key=lambda item: item[1])
            output_row = name_row.to_dict()
            center_y = float(name_row["center_y_norm"])
            output_row.update(
                {
                    "text": str(shirt_number),
                    "text_type": "shirt_number_candidate",
                    "score": round(score, 6),
                    "center_x_norm": round(number_center_x, 6),
                    "center_y_norm": round(center_y, 6),
                }
            )
            if {"x1", "x2", "y1", "y2", "center_x", "center_y"}.issubset(
                segment.columns
            ):
                center_x_pixels = number_center_x * width
                center_y_pixels = center_y * height
                output_row.update(
                    {
                        "x1": int(round(center_x_pixels - 20)),
                        "x2": int(round(center_x_pixels + 20)),
                        "y1": int(round(center_y_pixels - half_height)),
                        "y2": int(round(center_y_pixels + half_height)),
                        "center_x": round(center_x_pixels, 3),
                        "center_y": round(center_y_pixels, 3),
                    }
                )
            added_rows.append(output_row)

    if not added_rows:
        return segment, 0
    refined = pd.concat(
        [segment, pd.DataFrame(added_rows, columns=segment.columns)],
        ignore_index=True,
    )
    return refined, len(added_rows)


def formation_anchor_pairs(
    frame: pd.DataFrame,
) -> list[tuple[pd.Series, pd.Series]]:
    numbers = shirt_number_rows(frame)
    names = frame[
        (frame["text_type"] != "shirt_number_candidate")
        & frame["text"].map(is_table_name_like)
        & ~frame["text"].map(lambda value: parse_inline_player(value) is not None)
    ]
    anchors: list[tuple[pd.Series, pd.Series]] = []
    used_names: set[int] = set()
    for _, number_row in numbers.iterrows():
        candidates: list[tuple[float, int, pd.Series]] = []
        for name_index, name_row in names.iterrows():
            if int(name_index) in used_names:
                continue
            dx = abs(
                float(name_row["center_x_norm"])
                - float(number_row["center_x_norm"])
            )
            dy = (
                float(name_row["center_y_norm"])
                - float(number_row["center_y_norm"])
            )
            if dx <= 0.065 and 0.025 <= dy <= 0.13:
                candidates.append((dx + abs(dy - 0.06), int(name_index), name_row))
        if candidates:
            _, name_index, name_row = min(candidates, key=lambda item: item[0])
            used_names.add(name_index)
            anchors.append((number_row, name_row))
    return anchors


def formation_refinement_frames(
    segment: pd.DataFrame,
) -> list[tuple[int, list[tuple[pd.Series, pd.Series]]]]:
    candidates: list[tuple[int, list[tuple[pd.Series, pd.Series]]]] = []
    for frame_index, frame in segment.groupby("frame_index", sort=True):
        anchors = formation_anchor_pairs(frame)
        if len(anchors) >= 4:
            candidates.append((int(frame_index), anchors))
    if not candidates:
        return []
    maximum_anchors = max(len(item[1]) for item in candidates)
    best = [
        item for item in candidates if len(item[1]) == maximum_anchors
    ]
    best.sort(key=lambda item: item[0])
    positions = [0, (len(best) - 1) // 2, len(best) - 1]
    chosen: list[tuple[int, list[tuple[pd.Series, pd.Series]]]] = []
    for position in positions:
        candidate = best[position]
        if all(candidate[0] != existing[0] for existing in chosen):
            chosen.append(candidate)
    return chosen


def refine_formation_numbers(
    segment: pd.DataFrame,
    ocr: object,
    recognizer: object | None = None,
) -> tuple[pd.DataFrame, int]:
    chosen = formation_refinement_frames(segment)
    if not chosen or "frame_path" not in segment.columns:
        return segment, 0

    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise LineupResolutionError(
            "OpenCV is required for local formation OCR."
        ) from exc

    added_rows: list[dict[str, object]] = []
    for frame_index, anchors in chosen:
        frame = segment[segment["frame_index"] == frame_index]
        frame_path = resolve_project_path(Path(str(frame["frame_path"].iloc[0])))
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        height, width = image.shape[:2]

        anchor_names = [name_row for _, name_row in anchors]
        x_values = [
            float(name_row["center_x_norm"]) for name_row in anchor_names
        ]
        y_values = [
            float(name_row["center_y_norm"]) for name_row in anchor_names
        ]
        name_rows: list[pd.Series] = []
        for _, row in frame.iterrows():
            if (
                row["text_type"] == "shirt_number_candidate"
                or not is_table_name_like(row["text"])
                or parse_inline_player(row["text"]) is not None
            ):
                continue
            x = float(row["center_x_norm"])
            y = float(row["center_y_norm"])
            if (
                min(x_values) - 0.09 <= x <= max(x_values) + 0.09
                and min(y_values) - 0.08 <= y <= max(y_values) + 0.40
                and y <= 0.82
            ):
                name_rows.append(row)
        if not name_rows:
            continue

        crops: list[np.ndarray] = []
        valid_rows: list[pd.Series] = []
        for row in name_rows:
            x = float(row["center_x_norm"])
            y = float(row["center_y_norm"])
            x1 = max(0, int(round((x - 0.020) * width)))
            x2 = min(width, int(round((x + 0.020) * width)))
            y1 = max(0, int(round((y - 0.088) * height)))
            y2 = min(height, int(round((y - 0.030) * height)))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2]
            crops.append(
                cv2.resize(
                    crop,
                    None,
                    fx=8,
                    fy=8,
                    interpolation=cv2.INTER_CUBIC,
                )
            )
            valid_rows.append(row)

        results = list(ocr.predict(crops))
        if len(results) != len(valid_rows):
            continue

        numeric_results: list[tuple[int, float] | None] = []
        for result in results:
            data = local_result_data(result)
            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])
            numeric = [
                (int(str(text).strip()), float(score))
                for text, score in zip(texts, scores, strict=False)
                if str(text).strip().isdigit()
                and 1 <= int(str(text).strip()) <= 99
                and float(score) >= 0.75
            ]
            numeric_results.append(
                max(numeric, key=lambda item: item[1]) if numeric else None
            )

        missing_indices = [
            index
            for index, numeric in enumerate(numeric_results)
            if numeric is None
        ]
        if missing_indices:
            fallback_crops: list[np.ndarray] = []
            for index in missing_indices:
                row = valid_rows[index]
                x = float(row["center_x_norm"])
                y = float(row["center_y_norm"])
                x1 = max(0, int((x - 0.022) * width))
                x2 = min(width, int((x + 0.022) * width))
                y1 = max(0, int((y - 0.080) * height))
                y2 = min(height, int((y - 0.035) * height))
                crop = image[y1:y2, x1:x2]
                fallback_crops.append(
                    cv2.resize(
                        crop,
                        None,
                        fx=7,
                        fy=7,
                        interpolation=cv2.INTER_CUBIC,
                    )
                )
            fallback_results = list(ocr.predict(fallback_crops))
            for index, result in zip(
                missing_indices,
                fallback_results,
                strict=True,
            ):
                data = local_result_data(result)
                texts = data.get("rec_texts", [])
                scores = data.get("rec_scores", [])
                numeric = [
                    (int(str(text).strip()), float(score))
                    for text, score in zip(texts, scores, strict=False)
                    if str(text).strip().isdigit()
                    and 1 <= int(str(text).strip()) <= 99
                    and float(score) >= 0.75
                ]
                if numeric:
                    numeric_results[index] = max(
                        numeric,
                        key=lambda item: item[1],
                    )

        missing_indices = [
            index
            for index, numeric in enumerate(numeric_results)
            if numeric is None
        ]
        if missing_indices and recognizer is not None:
            recognition_crops: list[np.ndarray] = []
            for index in missing_indices:
                row = valid_rows[index]
                x = float(row["center_x_norm"])
                y = float(row["center_y_norm"])
                x1 = max(0, int((x - 0.015) * width))
                x2 = min(width, int((x + 0.015) * width))
                y1 = max(0, int((y - 0.080) * height))
                y2 = min(height, int((y - 0.035) * height))
                crop = image[y1:y2, x1:x2]
                recognition_crops.append(
                    cv2.resize(
                        crop,
                        None,
                        fx=8,
                        fy=8,
                        interpolation=cv2.INTER_CUBIC,
                    )
                )
            recognition_results = list(
                recognizer.predict(recognition_crops)
            )
            for index, result in zip(
                missing_indices,
                recognition_results,
                strict=True,
            ):
                data = local_result_data(result)
                text = str(data.get("rec_text", "")).strip()
                score = float(data.get("rec_score", 0.0))
                if (
                    text.isdigit()
                    and 1 <= int(text) <= 99
                    and score >= 0.75
                ):
                    numeric_results[index] = (int(text), score)

        for name_row, numeric in zip(
            valid_rows,
            numeric_results,
            strict=True,
        ):
            if numeric is None:
                continue
            shirt_number, score = numeric
            center_x = float(name_row["center_x_norm"])
            center_y = float(name_row["center_y_norm"]) - 0.058
            output_row = name_row.to_dict()
            output_row.update(
                {
                    "text": str(shirt_number),
                    "text_type": "shirt_number_candidate",
                    "score": round(score, 6),
                    "center_x_norm": round(center_x, 6),
                    "center_y_norm": round(center_y, 6),
                }
            )
            if {"x1", "x2", "y1", "y2", "center_x", "center_y"}.issubset(
                segment.columns
            ):
                center_x_pixels = center_x * width
                center_y_pixels = center_y * height
                output_row.update(
                    {
                        "x1": int(round(center_x_pixels - 18)),
                        "x2": int(round(center_x_pixels + 18)),
                        "y1": int(round(center_y_pixels - 18)),
                        "y2": int(round(center_y_pixels + 18)),
                        "center_x": round(center_x_pixels, 3),
                        "center_y": round(center_y_pixels, 3),
                    }
                )
            added_rows.append(output_row)

    if not added_rows:
        return segment, 0
    refined = pd.concat(
        [segment, pd.DataFrame(added_rows, columns=segment.columns)],
        ignore_index=True,
    )
    return refined, len(added_rows)


def load_detections(input_csv: Path) -> pd.DataFrame:
    if not input_csv.is_file():
        raise LineupResolutionError(f"OCR CSV does not exist: {input_csv}")

    detections = pd.read_csv(input_csv)
    missing = sorted(REQUIRED_COLUMNS - set(detections.columns))
    if missing:
        raise LineupResolutionError(
            "OCR CSV is missing required columns: " + ", ".join(missing)
        )
    if detections.empty:
        raise LineupResolutionError(f"OCR CSV has no rows: {input_csv}")

    numeric_columns = [
        "segment_index",
        "segment_start_seconds",
        "segment_end_seconds",
        "frame_index",
        "timestamp_seconds",
        "score",
        "center_x_norm",
        "center_y_norm",
    ]
    for column in numeric_columns:
        detections[column] = pd.to_numeric(detections[column], errors="raise")

    values = detections[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise LineupResolutionError("OCR CSV contains non-finite numeric values.")
    if not detections["score"].between(0, 1).all():
        raise LineupResolutionError("OCR confidence must be between 0 and 1.")
    if not detections["center_x_norm"].between(0, 1).all():
        raise LineupResolutionError("center_x_norm must be between 0 and 1.")
    if not detections["center_y_norm"].between(0, 1).all():
        raise LineupResolutionError("center_y_norm must be between 0 and 1.")

    detections["text"] = detections["text"].astype(str).str.strip()
    return detections[detections["text"] != ""].copy()


def shirt_number_rows(group: pd.DataFrame) -> pd.DataFrame:
    numbers = group[group["text_type"] == "shirt_number_candidate"].copy()
    parsed = pd.to_numeric(numbers["text"], errors="coerce")
    return numbers[parsed.between(1, 99)].copy()


def spatial_distance(left: pd.Series, right: pd.Series) -> tuple[float, float]:
    return (
        abs(float(left["center_x_norm"]) - float(right["center_x_norm"])),
        abs(float(left["center_y_norm"]) - float(right["center_y_norm"])),
    )


def signature_similarity(
    left: list[pd.Series],
    right: list[pd.Series],
    x_tolerance: float = 0.045,
    y_tolerance: float = 0.065,
) -> float:
    if not left or not right:
        return 0.0

    matched_right: set[int] = set()
    matches = 0
    for left_row in left:
        best_index: int | None = None
        best_distance = math.inf
        for index, right_row in enumerate(right):
            if index in matched_right or str(left_row["text"]) != str(right_row["text"]):
                continue
            dx, dy = spatial_distance(left_row, right_row)
            if dx <= x_tolerance and dy <= y_tolerance:
                distance = dx + dy
                if distance < best_distance:
                    best_distance = distance
                    best_index = index
        if best_index is not None:
            matched_right.add(best_index)
            matches += 1

    return matches / max(1, min(len(left), len(right)))


def detect_formation_events(
    segment: pd.DataFrame,
    min_number_count: int,
    max_gap_seconds: float,
    signature_threshold: float,
) -> list[FormationEvent]:
    snapshot_candidates: list[tuple[int, float, list[pd.Series]]] = []
    numbers = shirt_number_rows(segment)

    for frame_index, frame in numbers.groupby("frame_index", sort=True):
        if len(frame) < min_number_count:
            continue
        timestamp = float(frame["timestamp_seconds"].iloc[0])
        signature = [row for _, row in frame.iterrows()]
        snapshot_candidates.append((int(frame_index), timestamp, signature))

    events: list[FormationEvent] = []
    for frame_index, timestamp, signature in snapshot_candidates:
        if not events:
            events.append(FormationEvent([frame_index], [timestamp], [signature]))
            continue

        current = events[-1]
        time_gap = timestamp - current.end_seconds
        best_similarity = max(
            signature_similarity(signature, existing)
            for existing in current.signatures
        )
        if time_gap <= max_gap_seconds and best_similarity >= signature_threshold:
            current.snapshot_frames.append(frame_index)
            current.snapshot_timestamps.append(timestamp)
            current.signatures.append(signature)
        else:
            events.append(FormationEvent([frame_index], [timestamp], [signature]))

    return events


def cluster_number_positions(
    event: FormationEvent,
    segment: pd.DataFrame,
    x_tolerance: float = 0.045,
    y_tolerance: float = 0.065,
) -> list[NumberCluster]:
    snapshot_numbers = shirt_number_rows(
        segment[segment["frame_index"].isin(event.snapshot_frames)]
    ).sort_values(["frame_index", "center_y_norm", "center_x_norm"])
    clusters: list[NumberCluster] = []

    for _, observation in snapshot_numbers.iterrows():
        candidates: list[tuple[float, NumberCluster]] = []
        for cluster in clusters:
            dx = abs(float(observation["center_x_norm"]) - cluster.center_x)
            dy = abs(float(observation["center_y_norm"]) - cluster.center_y)
            if dx <= x_tolerance and dy <= y_tolerance:
                candidates.append((dx + dy, cluster))
        if candidates:
            min(candidates, key=lambda item: item[0])[1].observations.append(observation)
        else:
            clusters.append(NumberCluster(observations=[observation]))

    return clusters


def nearby_formation_labels(
    cluster: NumberCluster,
    event: FormationEvent,
    segment: pd.DataFrame,
) -> list[tuple[str, float, int]]:
    text_rows = segment[
        segment["frame_index"].isin(event.snapshot_frames)
        & (segment["text_type"] != "shirt_number_candidate")
    ]
    labels: list[tuple[str, float, int]] = []

    for frame_index, frame in text_rows.groupby("frame_index", sort=False):
        candidates: list[tuple[float, pd.Series]] = []
        for _, row in frame.iterrows():
            if not is_name_like(row["text"]):
                continue
            dx = abs(float(row["center_x_norm"]) - cluster.center_x)
            dy = float(row["center_y_norm"]) - cluster.center_y
            if dx <= 0.07 and 0.03 <= dy <= 0.13:
                candidates.append((dx + abs(dy - 0.078), row))
        if candidates:
            row = min(candidates, key=lambda item: item[0])[1]
            labels.append((str(row["text"]), float(row["score"]), int(frame_index)))

    return labels


def consensus_text(
    observations: list[tuple[str, float, int]],
    fuzzy_threshold: float = 0.84,
) -> tuple[str, float, int]:
    if not observations:
        return "", 0.0, 0

    groups: list[list[tuple[str, float, int]]] = []
    for observation in observations:
        for group in groups:
            if similarity(observation[0], group[0][0]) >= fuzzy_threshold:
                group.append(observation)
                break
        else:
            groups.append([observation])

    best_group = max(
        groups,
        key=lambda group: (
            len({frame for _, _, frame in group}),
            sum(score for _, score, _ in group),
        ),
    )
    by_variant: defaultdict[str, list[tuple[float, int]]] = defaultdict(list)
    display_value: dict[str, str] = {}
    for text, score, frame in best_group:
        normalized = normalize_text(text)
        by_variant[normalized].append((score, frame))
        display_value.setdefault(normalized, text)

    best_variant = max(
        by_variant,
        key=lambda key: (
            len({frame for _, frame in by_variant[key]}),
            sum(score for score, _ in by_variant[key]),
        ),
    )
    scores = [score for score, _ in by_variant[best_variant]]
    evidence_frames = {
        frame for _, _, frame in best_group
    }
    return (
        display_value[best_variant],
        float(np.mean(scores)),
        len(evidence_frames),
    )


def cluster_number_consensus(
    cluster: NumberCluster,
) -> tuple[int, float, int]:
    by_number: defaultdict[int, list[pd.Series]] = defaultdict(list)
    for row in cluster.observations:
        by_number[int(str(row["text"]))].append(row)

    best_number = max(
        by_number,
        key=lambda number: (
            len({int(row["frame_index"]) for row in by_number[number]}),
            sum(float(row["score"]) for row in by_number[number]),
        ),
    )
    rows = by_number[best_number]
    return (
        best_number,
        float(np.mean([float(row["score"]) for row in rows])),
        len({int(row["frame_index"]) for row in rows}),
    )


def select_player_clusters(
    clusters: list[NumberCluster],
    event: FormationEvent,
    segment: pd.DataFrame,
    expected_players: int,
) -> list[NumberCluster]:
    for cluster in clusters:
        cluster.label_observations = nearby_formation_labels(cluster, event, segment)

    ranked = sorted(
        clusters,
        key=lambda cluster: (
            bool(cluster.label_observations),
            len({frame for _, _, frame in cluster.label_observations}),
            cluster.frame_count,
            sum(float(row["score"]) for row in cluster.observations),
        ),
        reverse=True,
    )
    selected = ranked[:expected_players]
    return sorted(selected, key=lambda cluster: (cluster.center_y, cluster.center_x))


def formation_rows(clusters: list[NumberCluster]) -> dict[int, int]:
    row_by_cluster: dict[int, int] = {}
    current_row = 0
    previous_y: float | None = None
    for cluster in clusters:
        if previous_y is None or cluster.center_y - previous_y > 0.09:
            current_row += 1
        row_by_cluster[id(cluster)] = current_row
        previous_y = cluster.center_y
    return row_by_cluster


def best_full_name(
    formation_label: str,
    event_detections: pd.DataFrame,
    other_formation_labels: set[str],
) -> tuple[str, float, int]:
    candidates: list[tuple[str, float, int]] = []
    label_normalized = normalize_text(formation_label)
    label_occurrences = event_detections[
        event_detections.apply(
            lambda row: (
                row["text_type"] != "shirt_number_candidate"
                and similarity(row["text"], formation_label) >= 0.82
            ),
            axis=1,
        )
    ]

    for _, label_row in label_occurrences.iterrows():
        frame_index = int(label_row["frame_index"])
        frame = event_detections[
            (event_detections["frame_index"] == frame_index)
            & (event_detections["text_type"] != "shirt_number_candidate")
        ]
        neighbors: list[tuple[float, pd.Series]] = []
        for _, row in frame.iterrows():
            text = str(row["text"])
            normalized = normalize_text(text)
            if normalized == normalize_text(label_row["text"]):
                continue
            if not is_name_like(text) or normalized in other_formation_labels:
                continue
            dx = abs(
                float(row["center_x_norm"])
                - float(label_row["center_x_norm"])
            )
            dy = abs(
                float(row["center_y_norm"])
                - float(label_row["center_y_norm"])
            )
            if dx <= 0.085 and 0.012 <= dy <= 0.09:
                neighbors.append((dy + 0.5 * dx, row))

        if not neighbors:
            continue
        neighbor = min(neighbors, key=lambda item: item[0])[1]
        ordered = sorted(
            [label_row, neighbor],
            key=lambda row: (
                float(row["center_y_norm"]),
                float(row["center_x_norm"]),
            ),
        )
        combined = " ".join(str(row["text"]).strip() for row in ordered)
        combined_normalized = normalize_text(combined)
        if label_normalized not in combined_normalized:
            continue
        if len(combined_normalized.split()) > 6:
            continue
        candidates.append(
            (
                combined,
                float(label_row["score"]) * float(neighbor["score"]),
                frame_index,
            )
        )

    # Some OCR engines split a hyphenated surname into multiple boxes, for
    # example "Keane", "Lewis-", "Potter". Search compact vertical text stacks
    # whose combined normalized text contains the formation label.
    for frame_index, frame in event_detections.groupby("frame_index", sort=False):
        name_rows = [
            row
            for _, row in frame.iterrows()
            if (
                row["text_type"] != "shirt_number_candidate"
                and is_name_like(row["text"])
                and (
                    normalize_text(row["text"]) not in other_formation_labels
                    or similarity(row["text"], formation_label) >= 0.82
                )
            )
        ]
        for group_size in (2, 3):
            for rows in combinations(name_rows, group_size):
                x_values = [float(row["center_x_norm"]) for row in rows]
                y_values = [float(row["center_y_norm"]) for row in rows]
                if max(x_values) - min(x_values) > 0.065:
                    continue
                ordered = sorted(rows, key=lambda row: float(row["center_y_norm"]))
                ordered_y = [float(row["center_y_norm"]) for row in ordered]
                if ordered_y[-1] - ordered_y[0] > 0.14:
                    continue
                if any(
                    right - left > 0.09
                    for left, right in zip(ordered_y, ordered_y[1:])
                ):
                    continue

                combined = " ".join(str(row["text"]).strip() for row in ordered)
                combined = re.sub(r"\s*-\s*", "-", combined)
                combined_normalized = normalize_text(combined)
                if label_normalized not in combined_normalized:
                    continue
                if len(combined_normalized) <= len(label_normalized):
                    continue
                candidates.append(
                    (
                        combined,
                        float(np.mean([float(row["score"]) for row in ordered])),
                        int(frame_index),
                    )
                )

    # Do not let two ways of constructing the same name in one frame inflate
    # its vote.
    deduplicated: dict[tuple[int, str], tuple[str, float, int]] = {}
    for candidate in candidates:
        key = (candidate[2], normalize_text(candidate[0]))
        existing = deduplicated.get(key)
        if existing is None or candidate[1] > existing[1]:
            deduplicated[key] = candidate
    candidates = list(deduplicated.values())

    if not candidates:
        fallback = label_occurrences
        confidence = (
            float(fallback["score"].mean()) if not fallback.empty else 0.0
        )
        evidence = int(fallback["frame_index"].nunique())
        return formation_label, confidence, evidence

    full_name, confidence, evidence = consensus_text(
        candidates,
        fuzzy_threshold=0.87,
    )
    if len(normalize_text(full_name)) <= len(label_normalized):
        return formation_label, confidence, evidence
    return full_name, confidence, evidence


def resolve_event(
    event: FormationEvent,
    segment: pd.DataFrame,
    event_end_seconds: float,
    lineup_index: int,
    expected_players: int,
) -> list[dict[str, object]]:
    clusters = cluster_number_positions(event, segment)
    selected = select_player_clusters(
        clusters,
        event,
        segment,
        expected_players=expected_players,
    )
    if len(selected) < expected_players:
        raise LineupResolutionError(
            f"Only {len(selected)} player slots found for lineup {lineup_index}; "
            f"expected {expected_players}."
        )

    row_indices = formation_rows(selected)
    selected = sorted(
        selected,
        key=lambda cluster: (row_indices[id(cluster)], cluster.center_x),
    )
    label_results = {
        id(cluster): consensus_text(cluster.label_observations)
        for cluster in selected
    }
    missing_labels = [
        cluster
        for cluster in selected
        if not label_results[id(cluster)][0]
    ]
    if missing_labels:
        raise LineupResolutionError(
            f"{len(missing_labels)} player slot(s) have no nearby name label "
            f"for lineup {lineup_index}."
        )

    event_detections = segment[
        (segment["timestamp_seconds"] >= event.start_seconds)
        & (segment["timestamp_seconds"] < event_end_seconds)
    ]
    normalized_labels = {
        normalize_text(label_results[id(cluster)][0])
        for cluster in selected
    }
    records: list[dict[str, object]] = []

    for slot_index, cluster in enumerate(selected, start=1):
        shirt_number, number_confidence, number_evidence = (
            cluster_number_consensus(cluster)
        )
        formation_label, label_confidence, label_evidence = label_results[id(cluster)]
        other_labels = normalized_labels - {normalize_text(formation_label)}
        player_name, name_confidence, name_evidence = best_full_name(
            formation_label,
            event_detections,
            other_formation_labels=other_labels,
        )
        pair_confidence = (
            number_confidence * label_confidence * max(name_confidence, 0.01)
        ) ** (1 / 3)
        records.append(
            {
                "video": str(segment["video"].iloc[0]),
                "segment_index": int(segment["segment_index"].iloc[0]),
                "lineup_index": lineup_index,
                "resolution_method": "formation",
                "formation_timestamp_seconds": event.start_seconds,
                "slot_index": slot_index,
                "row_index": row_indices[id(cluster)],
                "shirt_number": shirt_number,
                "formation_label": formation_label,
                "player_name": player_name,
                "number_confidence": round(number_confidence, 6),
                "label_confidence": round(label_confidence, 6),
                "name_confidence": round(name_confidence, 6),
                "pair_confidence": round(pair_confidence, 6),
                "number_evidence_frames": number_evidence,
                "label_evidence_frames": label_evidence,
                "full_name_evidence_frames": name_evidence,
                "slot_center_x_norm": round(cluster.center_x, 6),
                "slot_center_y_norm": round(cluster.center_y, 6),
            }
        )

    return records


def attempt_formation_resolution(
    segment: pd.DataFrame,
    expected_players: int,
    min_number_count: int,
    max_gap_seconds: float,
    signature_threshold: float,
) -> tuple[
    pd.DataFrame,
    list[FormationEvent],
    list[dict[str, object]],
    list[str],
]:
    formation_segment = detections_without_substitute_panel(segment)
    events = detect_formation_events(
        formation_segment,
        min_number_count=min_number_count,
        max_gap_seconds=max_gap_seconds,
        signature_threshold=signature_threshold,
    )
    segment_records: list[dict[str, object]] = []
    event_errors: list[str] = []
    if not events:
        return formation_segment, events, segment_records, event_errors

    segment_end = float(formation_segment["segment_end_seconds"].max())
    for index, event in enumerate(events):
        event_end = (
            events[index + 1].start_seconds
            if index + 1 < len(events)
            else segment_end + 1e-6
        )
        try:
            segment_records.extend(
                resolve_event(
                    event,
                    formation_segment,
                    event_end_seconds=event_end,
                    lineup_index=index + 1,
                    expected_players=expected_players,
                )
            )
        except LineupResolutionError as exc:
            event_errors.append(str(exc))
    return formation_segment, events, segment_records, event_errors


def resolve_all_lineups(
    detections: pd.DataFrame,
    expected_players: int,
    min_number_count: int,
    max_gap_seconds: float,
    signature_threshold: float,
    enable_local_ocr: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    local_ocr: object | None = None
    local_number_recognizer: object | None = None

    grouped = detections.groupby(["video", "segment_index"], sort=False)
    for (video, segment_index), segment in grouped:
        messages: list[str] = []
        table_records, table_count = resolve_table_layout(
            segment,
            expected_players=expected_players,
        )
        if table_records:
            records.extend(table_records)
            diagnostics.append(
                {
                    "video": video,
                    "segment_index": int(segment_index),
                    "status": "resolved",
                    "resolution_method": "table",
                    "resolved_players": len(table_records),
                    "message": "Complete repeated table/list consensus.",
                }
            )
            print(
                f"{video}, segment {int(segment_index)}: "
                f"{len(table_records)} players resolved from table/list"
            )
            continue

        if table_count:
            messages.append(
                f"table/list pass found {table_count}/{expected_players} players"
            )
        if (
            enable_local_ocr
            and 4 <= table_count < expected_players
            and "frame_path" in segment.columns
        ):
            try:
                if local_ocr is None:
                    print("Loading PaddleOCR for local table-number refinement...")
                    local_ocr = create_local_table_ocr()
                refined_segment, added_count = refine_table_numbers(
                    segment,
                    ocr=local_ocr,
                    expected_players=expected_players,
                )
                if added_count:
                    messages.append(
                        f"local table OCR added {added_count} number observations"
                    )
                    table_records, refined_table_count = resolve_table_layout(
                        refined_segment,
                        expected_players=expected_players,
                    )
                    table_count = max(table_count, refined_table_count)
                    segment = refined_segment
                if table_records:
                    for record in table_records:
                        record["resolution_method"] = "table+local_ocr"
                    records.extend(table_records)
                    diagnostics.append(
                        {
                            "video": video,
                            "segment_index": int(segment_index),
                            "status": "resolved",
                            "resolution_method": "table+local_ocr",
                            "resolved_players": len(table_records),
                            "message": "; ".join(messages),
                        }
                    )
                    print(
                        f"{video}, segment {int(segment_index)}: "
                        f"{len(table_records)} players resolved from table/list "
                        f"after local OCR"
                    )
                    continue
            except (
                ImportError,
                LineupResolutionError,
                ModuleNotFoundError,
                OSError,
                ValueError,
            ) as exc:
                messages.append(f"local table OCR unavailable: {exc}")

        (
            _,
            initial_events,
            initial_records,
            initial_errors,
        ) = attempt_formation_resolution(
            segment,
            expected_players=expected_players,
            min_number_count=min_number_count,
            max_gap_seconds=max_gap_seconds,
            signature_threshold=signature_threshold,
        )
        if initial_events and not initial_errors:
            records.extend(initial_records)
            diagnostics.append(
                {
                    "video": video,
                    "segment_index": int(segment_index),
                    "status": "resolved",
                    "resolution_method": "formation",
                    "resolved_players": len(initial_records),
                    "message": "; ".join(messages),
                }
            )
            print(
                f"{video}, segment {int(segment_index)}: "
                f"{len(initial_events)} formation(s), "
                f"{len(initial_records)} players resolved"
            )
            continue

        if (
            enable_local_ocr
            and "frame_path" in segment.columns
            and formation_refinement_frames(segment)
        ):
            try:
                if local_ocr is None:
                    print(
                        "Loading PaddleOCR for local formation-number "
                        "refinement..."
                    )
                    local_ocr = create_local_table_ocr()
                if local_number_recognizer is None:
                    local_number_recognizer = (
                        create_local_number_recognizer()
                    )
                refined_segment, added_count = refine_formation_numbers(
                    segment,
                    ocr=local_ocr,
                    recognizer=local_number_recognizer,
                )
                if added_count:
                    segment = refined_segment
                    messages.append(
                        f"local formation OCR added {added_count} "
                        "number observations"
                    )
            except (
                ImportError,
                LineupResolutionError,
                ModuleNotFoundError,
                OSError,
                ValueError,
            ) as exc:
                messages.append(f"local formation OCR unavailable: {exc}")

        (
            _,
            events,
            segment_records,
            event_errors,
        ) = attempt_formation_resolution(
            segment,
            expected_players=expected_players,
            min_number_count=min_number_count,
            max_gap_seconds=max_gap_seconds,
            signature_threshold=signature_threshold,
        )
        if not events:
            messages.append("no formation snapshot found")
            diagnostics.append(
                {
                    "video": video,
                    "segment_index": int(segment_index),
                    "status": "unresolved",
                    "resolution_method": "",
                    "resolved_players": 0,
                    "message": "; ".join(messages),
                }
            )
            print(
                f"{video}, segment {int(segment_index)}: unresolved "
                f"({'; '.join(messages)})"
            )
            continue

        print(
            f"{video}, segment {int(segment_index)}: "
            f"detected {len(events)} lineup formation(s)"
        )
        if segment_records:
            print(f"  resolved {len(segment_records)} player rows")
        for error in event_errors:
            print(f"  unresolved ({error})")

        resolution_method = (
            "formation+local_ocr"
            if any(
                message.startswith("local formation OCR added")
                for message in messages
            )
            else "formation"
        )
        for record in segment_records:
            record["resolution_method"] = resolution_method
        records.extend(segment_records)
        messages.extend(event_errors)
        diagnostics.append(
            {
                "video": video,
                "segment_index": int(segment_index),
                "status": "resolved" if segment_records else "unresolved",
                "resolution_method": (
                    resolution_method if segment_records else ""
                ),
                "resolved_players": len(segment_records),
                "message": "; ".join(messages),
            }
        )

    return records, diagnostics


def validate_args(args: argparse.Namespace) -> None:
    if args.players_per_lineup <= 0:
        raise LineupResolutionError("--players-per-lineup must be positive.")
    if args.min_number_count <= 0:
        raise LineupResolutionError("--min-number-count must be positive.")
    if args.same_lineup_gap_seconds < 0:
        raise LineupResolutionError(
            "--same-lineup-gap-seconds cannot be negative."
        )
    if not 0 <= args.signature_threshold <= 1:
        raise LineupResolutionError("--signature-threshold must be between 0 and 1.")


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        input_csv = resolve_project_path(args.input_csv)
        output_csv = resolve_project_path(args.output_csv)
        diagnostics_csv = (
            resolve_project_path(args.diagnostics_csv)
            if args.diagnostics_csv is not None
            else output_csv.with_name(
                f"{output_csv.stem}_diagnostics{output_csv.suffix}"
            )
        )
        detections = load_detections(input_csv)
        records, diagnostics = resolve_all_lineups(
            detections,
            expected_players=args.players_per_lineup,
            min_number_count=args.min_number_count,
            max_gap_seconds=args.same_lineup_gap_seconds,
            signature_threshold=args.signature_threshold,
            enable_local_ocr=not args.disable_local_ocr,
        )
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records, columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)
        diagnostics_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            diagnostics,
            columns=DIAGNOSTIC_COLUMNS,
        ).to_csv(diagnostics_csv, index=False)
        print(f"Saved {len(records)} resolved player rows to: {output_csv}")
        print(f"Saved resolver diagnostics to: {diagnostics_csv}")
        return 0
    except (LineupResolutionError, OSError, pd.errors.ParserError, ValueError) as exc:
        print(f"Lineup resolution failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
