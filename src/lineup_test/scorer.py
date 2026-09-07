from __future__ import annotations

import re
from typing import Sequence

from lineup_test.config import PipelineConfig

# Clean clock / timer patterns (e.g. 0:07, 1:57, 14:25, 45+2, 90')
CLOCK_PATTERN = re.compile(r"\b\d{1,2}:\d{2}(?:\+\d+)?\b|\b\d{1,2}\s*['’]\b")

# Match scores (e.g. 2-0, 0.0, 0 - 0, 1:0)
SCORE_PATTERN = re.compile(r"\b\d+\s*[-|:.–]\s*\d+\b")

# Calendar years (e.g. 2024, 2026)
YEAR_PATTERN = re.compile(r"\b20\d{2}\b")

# Tactical football formations (e.g. 4-3-3, 3-5-2, 4-2-3-1)
FORMATION_PATTERN = re.compile(r"\b[1-5](?:\s*[-–]\s*[1-5]){2,3}\b")

# Jersey shirt numbers (1-99)
JERSEY_PATTERN = re.compile(r"\b(?:[1-9]|[1-9]\d)\b")

# Starting lineup structural keywords
LINEUP_KEYWORDS = (
    "lineup",
    "line up",
    "starting xi",
    "starting eleven",
    "substitutes",
    "formation",
    "coach",
    "doi hinh",
    "xuat phat",
    "titulares",
    "starters",
)

# Strictly non-lineup graphics terms (referees, aggregate score banners, match stats)
NEGATIVE_TERMS = (
    "aggregate",
    "referee",
    "match official",
    "officials",
    "standings",
    "possession",
    "attempts",
    "yellow card",
    "red card",
    "heineken",
    "recent matches",
    "group standings",
)

# Common broadcast, competition, sponsor, and role words.  These are useful OCR
# detections, but they are not evidence of player-name density.
NON_PLAYER_TOKENS = {
    "afc",
    "airways",
    "attempts",
    "booking",
    "bundesliga",
    "champions",
    "chartered",
    "coach",
    "cup",
    "defender",
    "elite",
    "essential",
    "europa",
    "expedia",
    "fifa",
    "final",
    "finals",
    "first",
    "football",
    "free",
    "formation",
    "goalkeeper",
    "group",
    "half",
    "head",
    "heineken",
    "highlights",
    "hotel",
    "hotels",
    "league",
    "lineup",
    "live",
    "match",
    "midfielder",
    "neom",
    "official",
    "officials",
    "qatar",
    "referee",
    "round",
    "semi-final",
    "sponsor",
    "stadium",
    "standard",
    "starters",
    "starting",
    "substitutes",
    "thank",
    "thanks",
    "titulares",
    "travel",
    "tv360",
    "uefa",
    "visa",
    "volunteers",
    "welcome",
    "world",
}


def _candidate_name_tokens(texts: Sequence[str]) -> list[str]:
    """Return unique name-like OCR tokens after removing broadcast vocabulary."""
    candidates: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for raw_word in text.split():
            word = raw_word.strip(".,:;!?()[]{}<>|/\\\"“”‘’")
            normalized = word.casefold().strip("-'_")
            if (
                len(normalized) < 3
                or any(char.isdigit() for char in normalized)
                or not any(char.isalpha() for char in normalized)
                or normalized in NON_PLAYER_TOKENS
                or ".com" in normalized
                or normalized.startswith("www.")
                or normalized in seen
            ):
                continue
            seen.add(normalized)
            candidates.append(word)
    return candidates


def compute_lineup_score(
    texts: Sequence[str],
    scores: Sequence[float],
    box_count: int,
    config: PipelineConfig | None = None,
) -> tuple[float, dict[str, object]]:
    """Compute discriminative lineup score without requiring spatial box adjacency.

    Filters match clocks, live gameplay scoreboards, referee cards, and sponsor bumpers.
    """
    cfg = config or PipelineConfig()
    clean_texts = [
        t.strip() for t, s in zip(texts, scores) if s >= cfg.min_box_confidence and t.strip()
    ]
    if not clean_texts:
        return 0.0, {
            "names": 0,
            "numbers": 0,
            "formation": False,
            "keyword": False,
            "strong_evidence": False,
            "suppressed": False,
            "sample_names": [],
            "sample_numbers": [],
        }

    full_text = " ".join(t.casefold() for t in clean_texts)

    # 1. Negative term suppression (referees, aggregate score banners, match statistics)
    neg_hits = [term for term in NEGATIVE_TERMS if term in full_text]
    if neg_hits:
        return 0.0, {
            "names": 0,
            "numbers": 0,
            "formation": False,
            "keyword": False,
            "strong_evidence": False,
            "suppressed": True,
            "reason": f"Negative terms detected: {neg_hits}",
            "sample_names": [],
            "sample_numbers": [],
        }

    has_formation = bool(FORMATION_PATTERN.search(full_text))
    has_keyword = any(keyword in full_text for keyword in LINEUP_KEYWORDS)

    # 2. Check for active match clock (e.g. 0:07, 1:57, 14:25)
    # Starting lineup presentations are broadcast before kickoff.
    # Running timers indicate live gameplay or scoreboard overlays.
    has_match_clock = bool(CLOCK_PATTERN.search(full_text))

    # 3. Extract genuine jersey numbers (after scrubbing clocks, scores, years)
    text_no_clocks = CLOCK_PATTERN.sub(" ", full_text)
    text_no_formations = FORMATION_PATTERN.sub(" ", text_no_clocks)
    text_no_scores = SCORE_PATTERN.sub(" ", text_no_formations)
    text_no_years = YEAR_PATTERN.sub(" ", text_no_scores)
    jersey_numbers = list(dict.fromkeys(JERSEY_PATTERN.findall(text_no_years)))

    # 4. Extract unique player-name candidates while excluding broadcast vocabulary.
    candidate_names = _candidate_name_tokens(clean_texts)

    # If an active match clock is present during gameplay, heavily suppress
    if has_match_clock and not (has_formation or has_keyword):
        return 0.0, {
            "names": len(candidate_names),
            "numbers": len(jersey_numbers),
            "formation": False,
            "keyword": False,
            "strong_evidence": False,
            "suppressed": True,
            "reason": "Active match clock detected (gameplay / scoreboard)",
            "sample_names": candidate_names[:6],
            "sample_numbers": jersey_numbers[:6],
        }

    # 5. Strict player name density requirement
    # Lineups contain 11 starting players; scoreboards only contain 2 team codes
    if len(candidate_names) < cfg.min_candidate_names and not has_formation:
        return (0.05 if len(candidate_names) >= 2 else 0.0), {
            "names": len(candidate_names),
            "numbers": len(jersey_numbers),
            "formation": has_formation,
            "keyword": has_keyword,
            "strong_evidence": False,
            "suppressed": False,
            "reason": "Insufficient player name candidates",
            "sample_names": candidate_names[:6],
            "sample_numbers": jersey_numbers[:6],
        }

    # 6. Multi-factor scoring (spatial layout agnostic)
    name_factor = min(1.0, len(candidate_names) / cfg.name_saturation_count)
    number_factor = min(1.0, len(jersey_numbers) / cfg.number_saturation_count)
    structure_factor = cfg.structure_bonus if (has_formation or has_keyword) else 0.0

    score = name_factor * 0.45 + number_factor * 0.35 + structure_factor
    if (
        not has_formation
        and not has_keyword
        and len(jersey_numbers) < cfg.min_unstructured_numbers
    ):
        score = min(score, cfg.unstructured_score_cap)
    score = min(1.0, score)

    strong_evidence = bool(
        has_formation
        or has_keyword
        or (
            len(candidate_names) >= cfg.strong_name_count
            and len(jersey_numbers) >= cfg.strong_number_count
            and box_count >= cfg.strong_name_count
        )
    )

    details = {
        "names": len(candidate_names),
        "numbers": len(jersey_numbers),
        "formation": has_formation,
        "keyword": has_keyword,
        "strong_evidence": strong_evidence,
        "box_count": box_count,
        "suppressed": False,
        "sample_names": candidate_names[:6],
        "sample_numbers": jersey_numbers[:6],
    }
    return round(score, 4), details
