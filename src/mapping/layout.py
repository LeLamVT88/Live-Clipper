from __future__ import annotations

import math

import numpy as np

from mapping.assignment import pair_panel
from mapping.schema import BBox, FrameAnalysis, OCRToken, Panel
from mapping.text import (
    COACH_WORDS, SUBSTITUTE_WORDS, is_anchor, is_player_name, names_compatible,
    parse_jersey_number,
)


def _bbox(points: object, width: int, height: int) -> BBox:
    array = np.asarray(points, dtype=float).reshape(-1, 2)
    if array.size == 0 or width <= 0 or height <= 0:
        raise ValueError("OCR polygon and frame dimensions must be non-empty")
    return (
        float(np.clip(array[:, 0].min() / width, 0, 1)),
        float(np.clip(array[:, 1].min() / height, 0, 1)),
        float(np.clip(array[:, 0].max() / width, 0, 1)),
        float(np.clip(array[:, 1].max() / height, 0, 1)),
    )


def make_tokens(texts: list[str], scores: list[float], polygons: list, frame_shape: tuple[int, ...]) -> list[OCRToken]:
    if not len(texts) == len(scores) == len(polygons):
        raise ValueError("OCR text/confidence/polygon counts do not match")
    height, width = frame_shape[:2]
    return [OCRToken(i, str(text).strip(), float(score), _bbox(box, width, height))
            for i, (text, score, box) in enumerate(zip(texts, scores, polygons))
            if str(text).strip()]


def _contains(region: BBox, token: OCRToken) -> bool:
    x, y = token.center
    return region[0] <= x <= region[2] and region[1] <= y <= region[3]


def _semantic_exclusions(tokens: list[OCRToken]) -> tuple[list[BBox], list[Panel]]:
    regions: list[BBox] = []
    panels: list[Panel] = []
    for token in tokens:
        x, _ = token.center
        if is_anchor(token.text, SUBSTITUTE_WORDS):
            if x < .45:
                related = [item for item in tokens if item.box[1] >= token.box[1]
                           and item.box[0] <= token.box[2] + .05 and item.center[0] < .42]
                boundary = max([token.box[2], *(item.box[2] for item in related)]) + .03
                region = (0.0, max(0.0, token.box[1] - .02), min(.48, boundary), 1.0)
            elif x > .55:
                related = [item for item in tokens if item.box[1] >= token.box[1]
                           and item.box[2] >= token.box[0] - .05 and item.center[0] > .58]
                boundary = min([token.box[0], *(item.box[0] for item in related)]) - .03
                region = (max(.52, boundary), max(0.0, token.box[1] - .02), 1.0, 1.0)
            else:
                region = (max(0.0, token.box[0] - .25), max(0.0, token.box[1] - .02),
                          min(1.0, token.box[2] + .25), 1.0)
            regions.append(region)
            panels.append(Panel("substitutes", region, [], {token.index}))
        elif is_anchor(token.text, COACH_WORDS):
            related = [item for item in tokens if token.box[1] <= item.box[1] <= token.box[3] + .14
                       and not (item.box[2] < token.box[0] - .04 or item.box[0] > token.box[2] + .04)]
            region = (max(0.0, min([token.box[0], *(item.box[0] for item in related)]) - .03),
                      max(0.0, token.box[1] - .02),
                      min(1.0, max([token.box[2], *(item.box[2] for item in related)]) + .03),
                      min(1.0, max([token.box[3], *(item.box[3] for item in related)]) + .03))
            regions.append(region)
            panels.append(Panel("coach", region, [], {token.index}))
    return regions, panels


def _expanded_box(tokens: list[OCRToken], role: str) -> BBox:
    x0 = min(t.box[0] for t in tokens)
    y0 = min(t.box[1] for t in tokens)
    x1 = max(t.box[2] for t in tokens)
    y1 = max(t.box[3] for t in tokens)
    if role == "starter_list":
        return max(0.0, x0 - .18), max(0.0, y0 - .02), min(1.0, x1 + .04), min(1.0, y1 + .02)
    return max(0.0, x0 - .08), max(0.0, y0 - .22), min(1.0, x1 + .08), min(1.0, y1 + .03)


def _best_list(names: list[OCRToken], all_tokens: list[OCRToken]) -> list[OCRToken]:
    best: tuple[float, list[OCRToken]] = (-math.inf, [])
    number_tokens = [token for token in all_tokens if parse_jersey_number(token.text) is not None]
    for anchor in names:
        column = sorted((t for t in names if abs(t.box[0] - anchor.box[0]) <= .065),
                        key=lambda t: t.center[1])
        # Drop duplicate OCR boxes occupying the same row.
        rows: list[OCRToken] = []
        for token in column:
            if not rows or token.center[1] - rows[-1].center[1] > .012:
                rows.append(token)
            elif token.confidence > rows[-1].confidence:
                rows[-1] = token
        for length in range(7, min(13, len(rows)) + 1):
            for start in range(len(rows) - length + 1):
                group = rows[start:start + length]
                gaps = np.diff([t.center[1] for t in group])
                median = float(np.median(gaps))
                if median <= .012 or median > .10 or float(gaps.max()) > median * 2.25:
                    continue
                regularity = float(np.std(gaps) / max(median, 1e-6))
                number_support = sum(any(
                    abs(number.center[1] - token.center[1]) <= max(.035, token.height * 1.2)
                    and -.025 <= token.box[0] - number.box[2] <= .18
                    for number in number_tokens) for token in group)
                mean_letters = float(np.mean([
                    sum(character.isalpha() for character in token.text) for token in group
                ]))
                score = (4.0 - abs(length - 11) * .35 - regularity
                         + number_support * .18 + min(.5, mean_letters * .035))
                if score > best[0]:
                    best = score, group
    return best[1]


def _pitch_rows(names: list[OCRToken]) -> list[list[OCRToken]]:
    rows: list[list[OCRToken]] = []
    for token in sorted(names, key=lambda t: (t.center[1], t.center[0])):
        target = next((row for row in rows if abs(np.mean([x.center[1] for x in row]) - token.center[1]) <= .042), None)
        if target is None:
            rows.append([token])
        else:
            target.append(token)
    return rows


def _best_pitch(names: list[OCRToken], all_tokens: list[OCRToken]) -> list[OCRToken]:
    # Large headings and edge watermarks should not enter the pitch structure.
    useful = [t for t in names if t.center[1] >= .10 and t.height <= .075
              and not (t.width > .34 and t.center[1] < .22)]
    number_tokens = [token for token in all_tokens if parse_jersey_number(token.text) is not None]

    def supported(name: OCRToken) -> bool:
        nx, _ = name.center
        return any(abs(number.center[0] - nx) <= max(.07, name.width * .75)
                   and -.025 <= name.box[1] - number.box[3] <= .22 for number in number_tokens)

    supported_names = [token for token in useful if supported(token)]
    if len(supported_names) >= 7:
        min_x = min(token.center[0] for token in supported_names) - .10
        max_x = max(token.center[0] for token in supported_names) + .10
        useful = [token for token in useful if supported(token) or min_x <= token.center[0] <= max_x]
    rows = _pitch_rows(useful)

    best: tuple[float, list[OCRToken]] = (-math.inf, [])
    for first in range(len(rows)):
        for last in range(first + 2, min(len(rows), first + 6)):
            selected_rows = rows[first:last + 1]
            group = [token for row in selected_rows for token in row]
            count = len(group)
            if not 7 <= count <= 13 or sum(len(row) >= 2 for row in selected_rows) < 2:
                continue
            xs = [token.center[0] for token in group]
            ys = [token.center[1] for token in group]
            if max(xs) - min(xs) < .22 or max(ys) - min(ys) < .18:
                continue
            support_count = sum(supported(token) for token in group)
            score = (5.0 - abs(count - 11) * .45 + sum(len(row) >= 2 for row in selected_rows) * .1
                     + support_count * .12)
            if score > best[0]:
                best = score, sorted(group, key=lambda t: (t.center[1], t.center[0]))
    return best[1]


def _overlap_ratio(left: list[OCRToken], right: list[OCRToken]) -> float:
    if not left or not right:
        return 0.0
    matches = sum(any(names_compatible(a.text, b.text) for b in right) for a in left)
    return matches / min(len(left), len(right))


def analyze_layout(
    texts: list[str], scores: list[float], polygons: list, frame_shape: tuple[int, ...],
    timestamp: float, sharpness: float,
) -> FrameAnalysis:
    tokens = make_tokens(texts, scores, polygons, frame_shape)
    exclusions, semantic_panels = _semantic_exclusions(tokens)
    names = [token for token in tokens if token.confidence >= .45 and is_player_name(token.text)
             and not any(_contains(region, token) for region in exclusions)]

    list_names = _best_list(names, tokens)
    list_indexes = {token.index for token in list_names}
    pitch_names = _best_pitch([token for token in names if token.index not in list_indexes], tokens)

    panels = list(semantic_panels)
    if pitch_names:
        panels.append(Panel("starter_pitch", _expanded_box(pitch_names, "starter_pitch"),
                            pitch_names, {t.index for t in pitch_names}))
    if list_names:
        role = "starter_list" if not pitch_names or _overlap_ratio(list_names, pitch_names) >= .60 else "auxiliary_list"
        panels.append(Panel(role, _expanded_box(list_names, "starter_list"),
                            list_names, {t.index for t in list_names}))

    observations = []
    for panel in panels:
        if panel.role.startswith("starter_"):
            observations.extend(pair_panel(panel, tokens, timestamp))

    unique_names = []
    for observation in sorted(observations, key=lambda o: o.name_confidence, reverse=True):
        if not any(names_compatible(observation.name, existing.name) for existing in unique_names):
            unique_names.append(observation)
    paired = sum(observation.jersey_number is not None for observation in unique_names)
    count = len(unique_names)
    issues = []
    if not panels or not observations:
        issues.append("no_starter_panel")
    if observations and not 7 <= count <= 13:
        issues.append(f"candidate_player_count_{count}")
    score = max(0.0, 1.0 - abs(count - 11) / 11) + min(1.0, paired / 11) + min(.2, sharpness / 5000)
    return FrameAnalysis(timestamp, panels, observations, sharpness, score, issues, {
        "texts": texts,
        "confidences": [round(float(value), 5) for value in scores],
        "polygons": polygons,
        "frame_shape": list(frame_shape),
    })
