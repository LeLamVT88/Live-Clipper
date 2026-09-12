from __future__ import annotations

from difflib import SequenceMatcher

from mapping.schema import BBox, ExclusionRegion, OCRToken, Panel
from mapping.text import COACH_WORDS, COMMENTATOR_WORDS, SUBSTITUTE_WORDS, normalized_text


def contains(region: BBox, token: OCRToken) -> bool:
    x, y = token.center
    return region[0] <= x <= region[2] and region[1] <= y <= region[3]


def _matches_anchor(value: str, anchors: tuple[str, ...]) -> bool:
    text = normalized_text(value)
    if text in anchors:
        return True
    return any(
        min(len(text), len(anchor)) >= 5
        and SequenceMatcher(None, text, anchor).ratio() >= .78
        for anchor in anchors
    )


def _substitutes_region(anchor: OCRToken, tokens: list[OCRToken]) -> BBox:
    x, _ = anchor.center
    if x < .45:
        related = [token for token in tokens if token.box[1] >= anchor.box[1]
                   and token.box[0] <= anchor.box[2] + .05 and token.center[0] < .42]
        boundary = max([anchor.box[2], *(token.box[2] for token in related)]) + .03
        return 0.0, max(0.0, anchor.box[1] - .02), min(.48, boundary), 1.0
    if x > .55:
        related = [token for token in tokens if token.box[1] >= anchor.box[1]
                   and token.box[2] >= anchor.box[0] - .05 and token.center[0] > .58]
        boundary = min([anchor.box[0], *(token.box[0] for token in related)]) - .03
        return max(.52, boundary), max(0.0, anchor.box[1] - .02), 1.0, 1.0
    return (max(0.0, anchor.box[0] - .25), max(0.0, anchor.box[1] - .02),
            min(1.0, anchor.box[2] + .25), 1.0)


def _coach_region(anchor: OCRToken, tokens: list[OCRToken]) -> BBox:
    related = [token for token in tokens if anchor.box[1] <= token.box[1] <= anchor.box[3] + .14
               and not (token.box[2] < anchor.box[0] - .04
                        or token.box[0] > anchor.box[2] + .04)]
    return (
        max(0.0, min([anchor.box[0], *(token.box[0] for token in related)]) - .03),
        max(0.0, anchor.box[1] - .02),
        min(1.0, max([anchor.box[2], *(token.box[2] for token in related)]) + .03),
        min(1.0, max([anchor.box[3], *(token.box[3] for token in related)]) + .03),
    )


def _commentator_region(anchor: OCRToken, tokens: list[OCRToken]) -> BBox:
    related = [token for token in tokens
               if anchor.box[1] - .10 <= token.center[1] <= anchor.box[3] + .14
               and not (token.box[2] < anchor.box[0] - .35
                        or token.box[0] > anchor.box[2] + .35)]
    return (
        max(0.0, min([anchor.box[0], *(token.box[0] for token in related)]) - .04),
        max(0.0, min([anchor.box[1], *(token.box[1] for token in related)]) - .03),
        min(1.0, max([anchor.box[2], *(token.box[2] for token in related)]) + .04),
        min(1.0, max([anchor.box[3], *(token.box[3] for token in related)]) + .03),
    )


def detect_exclusions(tokens: list[OCRToken]) -> tuple[list[ExclusionRegion], list[Panel]]:
    """Find semantic regions before any player-card inference or number assignment."""
    exclusions: list[ExclusionRegion] = []
    panels: list[Panel] = []
    for token in tokens:
        if _matches_anchor(token.text, SUBSTITUTE_WORDS):
            box = _substitutes_region(token, tokens)
            exclusions.append(ExclusionRegion("substitutes", box, token.text))
            panels.append(Panel("substitutes", box, [], {token.index}))
        elif _matches_anchor(token.text, COACH_WORDS):
            box = _coach_region(token, tokens)
            exclusions.append(ExclusionRegion("coach", box, token.text))
            panels.append(Panel("coach", box, [], {token.index}))
        elif _matches_anchor(token.text, COMMENTATOR_WORDS):
            box = _commentator_region(token, tokens)
            exclusions.append(ExclusionRegion("commentators", box, token.text))
            panels.append(Panel("commentators", box, [], {token.index}))
    return exclusions, panels


def outside_exclusions(tokens: list[OCRToken], exclusions: list[ExclusionRegion]) -> list[OCRToken]:
    return [token for token in tokens
            if not any(contains(exclusion.box, token) for exclusion in exclusions)]
