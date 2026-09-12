from __future__ import annotations

import math

import numpy as np

from mapping.schema import BBox, OCRToken, Panel, PlayerCard, PlayerObservation
from mapping.text import names_compatible, parse_jersey_number


def _expanded_box(tokens: list[OCRToken], role: str) -> BBox:
    x0, y0 = min(token.box[0] for token in tokens), min(token.box[1] for token in tokens)
    x1, y1 = max(token.box[2] for token in tokens), max(token.box[3] for token in tokens)
    if role == "starter_list":
        return max(0.0, x0 - .18), max(0.0, y0 - .02), min(1.0, x1 + .04), min(1.0, y1 + .02)
    return max(0.0, x0 - .08), max(0.0, y0 - .22), min(1.0, x1 + .08), min(1.0, y1 + .03)


def _best_list(names: list[OCRToken], all_tokens: list[OCRToken]) -> list[OCRToken]:
    best: tuple[float, list[OCRToken]] = (-math.inf, [])
    numbers = [token for token in all_tokens if parse_jersey_number(token.text) is not None]
    for anchor in names:
        column = sorted((token for token in names if abs(token.box[0] - anchor.box[0]) <= .065),
                        key=lambda token: token.center[1])
        rows: list[OCRToken] = []
        for token in column:
            if not rows or token.center[1] - rows[-1].center[1] > .012:
                rows.append(token)
            elif token.confidence > rows[-1].confidence:
                rows[-1] = token
        for length in range(7, min(13, len(rows)) + 1):
            for start in range(len(rows) - length + 1):
                group = rows[start:start + length]
                gaps = np.diff([token.center[1] for token in group])
                median_gap = float(np.median(gaps))
                if median_gap <= .012 or median_gap > .10 or float(gaps.max()) > median_gap * 2.25:
                    continue
                regularity = float(np.std(gaps) / max(median_gap, 1e-6))
                support = sum(any(
                    abs(number.center[1] - token.center[1]) <= max(.035, token.height * 1.2)
                    and -.025 <= token.box[0] - number.box[2] <= .18
                    for number in numbers) for token in group)
                letters = np.mean([sum(character.isalpha() for character in token.text)
                                   for token in group])
                score = 4.0 - abs(length - 11) * .35 - regularity + support * .18 + min(.5, letters * .035)
                if score > best[0]:
                    best = score, group
    return best[1]


def _pitch_rows(names: list[OCRToken]) -> list[list[OCRToken]]:
    rows: list[list[OCRToken]] = []
    for token in sorted(names, key=lambda item: (item.center[1], item.center[0])):
        row = next((items for items in rows
                    if abs(np.mean([item.center[1] for item in items]) - token.center[1]) <= .042), None)
        if row is None:
            rows.append([token])
        else:
            row.append(token)
    return rows


def _best_pitch(
    names: list[OCRToken], all_tokens: list[OCRToken], *, min_number_support: int,
) -> list[OCRToken]:
    numbers = [token for token in all_tokens if parse_jersey_number(token.text) is not None]

    def supported(name: OCRToken) -> bool:
        return any(abs(number.center[0] - name.center[0]) <= max(.07, name.width * .75)
                   and -.025 <= name.box[1] - number.box[3] <= .22 for number in numbers)

    useful = [token for token in names
              if token.height <= .075
              and not (token.width > .34 and token.center[1] < .22)
              and not (not supported(token) and token.center[1] < .155)
              and not (not supported(token) and token.center[1] < .24
                       and (token.center[0] < .13 or token.center[0] > .87))]
    supported_names = [token for token in useful if supported(token)]
    if len(supported_names) >= 7:
        min_x = min(token.center[0] for token in supported_names) - .10
        max_x = max(token.center[0] for token in supported_names) + .10
        useful = [token for token in useful if supported(token) or min_x <= token.center[0] <= max_x]

    best: tuple[float, list[OCRToken]] = (-math.inf, [])
    rows = _pitch_rows(useful)
    for first in range(len(rows)):
        for last in range(first + 2, min(len(rows), first + 6)):
            selected_rows = rows[first:last + 1]
            group = [token for row in selected_rows for token in row]
            if not 7 <= len(group) <= 13 or sum(len(row) >= 2 for row in selected_rows) < 2:
                continue
            xs, ys = [token.center[0] for token in group], [token.center[1] for token in group]
            if max(xs) - min(xs) < .22 or max(ys) - min(ys) < .18:
                continue
            support = sum(supported(token) for token in group)
            if support < min_number_support:
                continue
            score = 5.0 - abs(len(group) - 11) * .45 + sum(
                len(row) >= 2 for row in selected_rows) * .1 + support * .12
            if score > best[0]:
                best = score, sorted(group, key=lambda token: (token.center[1], token.center[0]))
    return best[1]


def _overlap(left: list[OCRToken], right: list[OCRToken]) -> float:
    if not left or not right:
        return 0.0
    matches = sum(any(names_compatible(a.text, b.text) for b in right) for a in left)
    return matches / min(len(left), len(right))


def detect_player_panels(names: list[OCRToken], tokens: list[OCRToken]) -> list[Panel]:
    """Infer list/pitch structures after excluded regions have been removed."""
    list_names = _best_list(names, tokens)
    list_indexes = {token.index for token in list_names}
    pitch_pool = [token for token in names if token.index not in list_indexes]
    pitch_names = _best_pitch(pitch_pool, tokens, min_number_support=3)
    weak_pitch = [] if pitch_names else _best_pitch(pitch_pool, tokens, min_number_support=0)

    panels: list[Panel] = []
    if pitch_names:
        panels.append(Panel("starter_pitch", _expanded_box(pitch_names, "starter_pitch"),
                            pitch_names, {token.index for token in pitch_names}))
    elif weak_pitch:
        panels.append(Panel("weak_pitch", _expanded_box(weak_pitch, "starter_pitch"),
                            weak_pitch, {token.index for token in weak_pitch}))
    if list_names:
        role = "starter_list" if not pitch_names or _overlap(list_names, pitch_names) >= .60 else "auxiliary_list"
        panels.append(Panel(role, _expanded_box(list_names, "starter_list"),
                            list_names, {token.index for token in list_names}))
    return panels


def number_roi(name_box: BBox, role: str) -> BBox:
    x0, y0, x1, y1 = name_box
    if role == "starter_list":
        return max(0.0, x0 - .16), max(0.0, y0 - .018), max(0.0, x0 - .003), min(1.0, y1 + .018)
    center = (x0 + x1) / 2
    half_width = max(.028, min(.047, (x1 - x0) * .52))
    return max(0.0, center - half_width), max(0.0, y0 - .09), min(1.0, center + half_width), max(0.0, y0 - .001)


def build_player_cards(
    panels: list[Panel], observations: list[PlayerObservation],
) -> list[PlayerCard]:
    """Materialize visual slots so frame selection does not depend only on OCR pairs."""
    cards: list[PlayerCard] = []
    for panel in panels:
        if panel.role not in {"starter_pitch", "starter_list", "weak_pitch"}:
            continue
        for token in panel.name_tokens:
            observation = next((item for item in observations
                                if item.panel_role == panel.role
                                and names_compatible(item.name, token.text)), None)
            roi = observation.number_box if observation and observation.number_box else number_roi(token.box, panel.role)
            card_box = (
                min(token.box[0], roi[0]), min(token.box[1], roi[1]),
                max(token.box[2], roi[2]), max(token.box[3], roi[3]),
            )
            cards.append(PlayerCard(
                panel.role, token.text, card_box, token.box, roi, token.confidence,
                observation.jersey_number if observation else None,
            ))
    return cards
