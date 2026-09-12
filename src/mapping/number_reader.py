from __future__ import annotations

from collections import Counter
from statistics import median

import cv2
import numpy as np

from mapping.cards import number_roi
from mapping.schema import BBox, FrameAnalysis, PlayerObservation
from mapping.text import normalized_text, parse_number_crop
from ocr_engine import run_ocr_batch


VARIANTS_PER_CROP = 6


def _search_boxes(observation: PlayerObservation, anchors: list[PlayerObservation]) -> list[BBox]:
    boxes: list[BBox] = []
    usable = [anchor for anchor in anchors if anchor.panel_role == observation.panel_role
              and anchor.number_box is not None and anchor.number_source != "inline_ocr"]
    if len(usable) >= 2:
        relative_x = []
        relative_y = []
        widths = []
        heights = []
        for anchor in usable:
            name_x = (anchor.name_box[0] + anchor.name_box[2]) / 2
            name_y = (anchor.name_box[1] + anchor.name_box[3]) / 2
            number_x = (anchor.number_box[0] + anchor.number_box[2]) / 2
            number_y = (anchor.number_box[1] + anchor.number_box[3]) / 2
            relative_x.append(number_x - name_x)
            relative_y.append(number_y - name_y)
            widths.append(anchor.number_box[2] - anchor.number_box[0])
            heights.append(anchor.number_box[3] - anchor.number_box[1])
        name_x = (observation.name_box[0] + observation.name_box[2]) / 2
        name_y = (observation.name_box[1] + observation.name_box[3]) / 2
        center_x = name_x + median(relative_x)
        center_y = name_y + median(relative_y)
        half_width = max(.018, min(.047, median(widths) * 1.55))
        half_height = max(.022, min(.058, median(heights) * 1.3))
        boxes.append((max(0.0, center_x - half_width), max(0.0, center_y - half_height),
                      min(1.0, center_x + half_width), min(1.0, center_y + half_height)))
    fallback = number_roi(observation.name_box, observation.panel_role)
    if not boxes or max(abs(a - b) for a, b in zip(boxes[0], fallback)) > .015:
        boxes.append(fallback)
    return boxes


def _crop(frame: np.ndarray, box: BBox) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = (
        max(0, min(width, round(box[0] * width))),
        max(0, min(height, round(box[1] * height))),
        max(0, min(width, round(box[2] * width))),
        max(0, min(height, round(box[3] * height))),
    )
    patch = frame[y0:y1, x0:x1]
    return patch if patch.size and patch.shape[0] >= 4 and patch.shape[1] >= 4 else None


def _variants(patch: np.ndarray) -> list[np.ndarray]:
    scale = max(1.0, 120 / max(1, patch.shape[0]))
    enlarged = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blue = enlarged[:, :, 0]
    blue_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(blue)
    _, blue_otsu = cv2.threshold(blue, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [
        enlarged,
        cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(255 - otsu, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(blue_clahe, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(blue_otsu, cv2.COLOR_GRAY2BGR),
    ]


def recover_frame_numbers(frame: np.ndarray, analysis: FrameAnalysis, model: object) -> list[dict[str, object]]:
    """Recover missing digits locally; variants support one timestamp, not extra votes."""
    anchors = [observation for observation in analysis.starter_observations
               if observation.jersey_number is not None and observation.number_box is not None]
    pending: list[tuple[PlayerObservation, list[BBox], list[np.ndarray]]] = []
    batch: list[np.ndarray] = []
    for observation in analysis.starter_observations:
        if observation.jersey_number is not None:
            continue
        boxes = _search_boxes(observation, anchors)
        variants = []
        valid_boxes = []
        for box in boxes:
            patch = _crop(frame, box)
            if patch is None:
                continue
            valid_boxes.append(box)
            variants.extend(_variants(patch))
        if variants:
            pending.append((observation, valid_boxes, variants))
            batch.extend(variants)
    if not batch:
        return []

    results = run_ocr_batch(batch, model)
    cursor = 0
    evidence = []
    for observation, boxes, variants in pending:
        readings = []
        votes: Counter[int] = Counter()
        weighted_votes: Counter[int] = Counter()
        for variant_index in range(len(variants)):
            texts, scores, _ = results[cursor]
            cursor += 1
            value_scores: dict[int, float] = {}
            contains_name = any(sum(character.isalpha() for character in normalized_text(text)) >= 3
                                for text in texts)
            for text, score in zip(texts, scores):
                number = parse_number_crop(text)
                explicit_digits = normalized_text(text).isdigit()
                # A letter-like glyph is allowed only in an isolated number crop.
                # This prevents the L in RICCI/other nearby name text becoming 1.
                if number is not None and (explicit_digits or (not contains_name and float(score) >= .55)):
                    value_scores[number] = max(value_scores.get(number, 0.0), float(score))
            values = sorted(value_scores)
            readings.append({"crop": variant_index // VARIANTS_PER_CROP,
                             "variant": variant_index % VARIANTS_PER_CROP, "texts": texts,
                             "confidences": [round(float(score), 4) for score in scores], "numbers": values})
            if len(values) == 1:
                votes[values[0]] += 1
                weighted_votes[values[0]] += value_scores[values[0]]
        candidates = sorted(votes)
        observation.number_candidates = sorted(set(observation.number_candidates) | set(candidates))
        for candidate in candidates:
            observation.number_candidate_scores[candidate] = max(
                observation.number_candidate_scores.get(candidate, 0.0),
                float(weighted_votes[candidate] + votes[candidate] * .25),
            )
        ranked = sorted(
            ((votes[number], weighted_votes[number], number) for number in candidates),
            reverse=True,
        )
        accepted = None
        if ranked and ranked[0][0] >= 2:
            runner_votes, runner_weight = (ranked[1][0], ranked[1][1]) if len(ranked) > 1 else (0, 0.0)
            if (ranked[0][0] >= runner_votes + 1
                    and ranked[0][1] >= max(.01, runner_weight) * 1.25):
                accepted = ranked[0][2]
        if accepted is not None:
            observation.jersey_number = accepted
            observation.number_confidence = min(1.0, weighted_votes[accepted] / max(2, votes[accepted]))
            observation.number_box = boxes[0]
            observation.number_source = "local_preprocessed_ocr"
            observation.pair_confidence = observation.number_confidence * .8
        evidence.append({
            "timestamp": round(analysis.timestamp, 3), "name": observation.name,
            "search_box": [round(value, 5) for value in boxes[0]],
            "search_boxes": [[round(value, 5) for value in box] for box in boxes],
            "variant_readings": readings,
            "candidate_votes": {str(number): votes[number] for number in candidates},
            "candidate_scores": {
                str(number): round(float(weighted_votes[number]), 4) for number in candidates
            },
            "accepted_number": observation.jersey_number if observation.number_source == "local_preprocessed_ocr" else None,
        })
    return evidence


def recover_unresolved_from_frames(
    frames: list[tuple[float, np.ndarray]], players: list[dict[str, object]], model: object,
) -> tuple[list[FrameAnalysis], list[dict[str, object]]]:
    """OCR only missing number ROIs on nearby frames, without full-frame OCR."""
    unresolved = [player for player in players if not player["number_confirmed"]]
    analyses: list[FrameAnalysis] = []
    evidence: list[dict[str, object]] = []
    for timestamp, frame in frames:
        observations: list[PlayerObservation] = []
        for player in unresolved:
            reference = max(player["observations"], key=lambda item: item["name_confidence"])
            observations.append(PlayerObservation(
                timestamp=timestamp,
                panel_role=reference["panel_role"],
                name=player["name"],
                name_confidence=float(reference["name_confidence"]),
                name_box=tuple(reference["name_box"]),
            ))
        analysis = FrameAnalysis(timestamp, [], observations, 0.0, 0.0, [], {})
        evidence.extend(recover_frame_numbers(frame, analysis, model))
        analyses.append(analysis)
    return analyses, evidence
