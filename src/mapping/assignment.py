from __future__ import annotations

from functools import lru_cache

from mapping.schema import OCRToken, Panel, PlayerObservation
from mapping.text import clean_player_name, parse_jersey_number, parse_number_crop, split_inline_player


def _spatial_number(token: OCRToken) -> int | None:
    """Allow digit-like glyphs only after panel geometry identifies a number token."""
    return parse_jersey_number(token.text) or parse_number_crop(token.text)


def _inside(box: tuple[float, float, float, float], token: OCRToken, padding: float = .04) -> bool:
    x, y = token.center
    return box[0] - padding <= x <= box[2] + padding and box[1] - padding <= y <= box[3] + padding


def _edge_score(role: str, name: OCRToken, number: OCRToken) -> float | None:
    nx, ny = name.center
    dx, dy = number.center
    if role == "starter_list":
        vertical = abs(dy - ny)
        horizontal = name.box[0] - number.box[2]
        if vertical > max(.035, name.height * 1.2) or not -.025 <= horizontal <= .18:
            return None
        return 1.5 - 8 * vertical - 2.5 * max(0.0, horizontal)
    horizontal = abs(dx - nx)
    vertical = name.box[1] - number.box[3]
    if horizontal > max(.065, name.width * .7) or not -.025 <= vertical <= .15:
        return None
    return 1.4 - 6 * horizontal - 2.4 * max(0.0, vertical)


def _global_assignment(names: list[OCRToken], numbers: list[OCRToken], role: str) -> dict[int, int]:
    edges: list[list[tuple[int, float]]] = []
    for name in names:
        candidates = []
        for j, number in enumerate(numbers):
            score = _edge_score(role, name, number)
            if score is not None and score > .25:
                candidates.append((j, score * (.5 + .5 * number.confidence)))
        edges.append(sorted(candidates, key=lambda item: item[1], reverse=True)[:5])

    @lru_cache(maxsize=None)
    def solve(i: int, used: int) -> tuple[float, tuple[tuple[int, int], ...]]:
        if i == len(names):
            return 0.0, ()
        best_score, best_pairs = solve(i + 1, used)
        for j, score in edges[i]:
            bit = 1 << j
            if used & bit:
                continue
            tail_score, tail_pairs = solve(i + 1, used | bit)
            proposal = score + tail_score
            if proposal > best_score:
                best_score, best_pairs = proposal, ((i, j),) + tail_pairs
        return best_score, best_pairs

    return dict(solve(0, 0)[1])


def pair_panel(panel: Panel, tokens: list[OCRToken], timestamp: float) -> list[PlayerObservation]:
    names = panel.name_tokens
    inline: dict[int, tuple[int, str]] = {}
    for i, token in enumerate(names):
        number, clean_name = split_inline_player(token.text)
        if number is not None:
            inline[i] = (number, clean_name)

    number_tokens = [token for token in tokens if token.index not in panel.token_indexes
                     and _spatial_number(token) is not None and _inside(panel.box, token)]
    # Keep the DP bounded when scoreboards or tables leak into a panel.
    number_tokens = sorted(number_tokens, key=lambda token: token.confidence, reverse=True)[:16]
    remaining_names = [token for i, token in enumerate(names) if i not in inline]
    assignment = _global_assignment(remaining_names, number_tokens, panel.role)
    assigned_by_token = {token.index: number_tokens[j] for i, j in assignment.items()
                         for token in [remaining_names[i]]}

    result = []
    for i, token in enumerate(names):
        display_name = clean_player_name(inline.get(i, (None, token.text))[1])
        observation = PlayerObservation(
            timestamp=timestamp, panel_role=panel.role, name=display_name,
            name_confidence=token.confidence, name_box=token.box,
        )
        if i in inline:
            observation.jersey_number = inline[i][0]
            observation.number_candidates = [inline[i][0]]
            observation.number_candidate_scores = {inline[i][0]: token.confidence}
            observation.number_confidence = token.confidence
            observation.number_box = token.box
            observation.number_source = "inline_ocr"
            observation.pair_confidence = token.confidence
        elif token.index in assigned_by_token:
            number_token = assigned_by_token[token.index]
            number = _spatial_number(number_token)
            observation.jersey_number = number
            observation.number_candidates = [number] if number is not None else []
            observation.number_candidate_scores = ({number: number_token.confidence}
                                                   if number is not None else {})
            observation.number_confidence = number_token.confidence
            observation.number_box = number_token.box
            observation.number_source = "spatial_ocr"
            edge = _edge_score(panel.role, token, number_token) or 0.0
            observation.pair_confidence = min(1.0, max(0.0, edge / 1.5))
        result.append(observation)
    return result
