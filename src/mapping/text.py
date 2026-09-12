from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher


SUBSTITUTE_WORDS = ("SUBSTITUTE", "SUBSTITUTES", "SUBS", "BENCH", "RESERVES")
COACH_WORDS = ("HEAD COACH", "COACH", "MANAGER", "ENTRAINEUR", "TREINER")
COMMENTATOR_WORDS = (
    "COMMENTATOR", "COMMENTATORS", "COMMENTARY", "CASTER", "CASTERS",
    "BLV", "BINH LUAN VIEN",
)
CLUB_MARKERS = {"FC", "AFC", "CF", "SC", "LFC", "LOSC", "AS", "AC", "FK", "SK"}
OVERLAY_WORDS = {
    "TV", "LIVE", "DIRECT", "TRUC TIEP", "TRUC TIEP HD", "UEFA COM",
    "OFFICIAL", "MATCH OFFICIAL", "COMMENTARY", "COMMENTATOR",
}
NON_PLAYER_WORDS = {
    "LINEUP", "LINE UPS", "STARTING XI", "FORMATION", "TEAM FORMATION",
    "TEAM", "LEAGUE", "ELITE", "CHAMPIONS",
    "SERIE A", "PREMIER LEAGUE", "EMIRATES",
    "GOALKEEPER", "GK", "CAPTAIN", "REFEREE", "COMMENTATOR", "COMMENTARY",
    "LIVE", "DIRECT", "HEAD COACH", "COACH", "MANAGER", "SUBSTITUTES",
    "SUBSTITUTE", "SUBS", "BENCH", "RESERVES",
}


def normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^A-Z0-9]+", " ", value.upper()).strip()
    return re.sub(r"\s+", " ", value)


def name_key(value: str) -> str:
    return normalized_text(value).replace(" ", "")


def parse_jersey_number(value: str) -> int | None:
    cleaned = normalized_text(value)
    if not re.fullmatch(r"\d{1,2}", cleaned):
        return None
    number = int(cleaned)
    return number if 1 <= number <= 99 else None


def parse_number_crop(value: str) -> int | None:
    """Read OCR confusions only after geometry has isolated a number region."""
    cleaned = re.sub(r"[^A-Z0-9|]", "", normalized_text(value))
    if not cleaned or len(cleaned) > 2:
        return None
    translated = cleaned.translate(str.maketrans({"I": "1", "L": "1", "|": "1", "O": "0"}))
    return parse_jersey_number(translated)


def split_inline_player(value: str) -> tuple[int | None, str]:
    match = re.match(r"^\s*(\d{1,2})\s+(.+?)\s*$", value)
    if not match:
        return None, value.strip()
    number = int(match.group(1))
    return (number if 1 <= number <= 99 else None), match.group(2).strip()


def clean_player_name(value: str) -> str:
    """Remove presentation-only role tags while preserving the displayed spelling."""
    _, candidate = split_inline_player(value)
    candidate = re.sub(r"\s*[\[(](?:GK|C|CAPTAIN)[\])]\s*$", "", candidate,
                       flags=re.IGNORECASE).strip()
    return re.sub(r"\s+", " ", candidate)


def is_likely_metadata(value: str) -> bool:
    """Reject generic club/broadcast labels without maintaining a team-name list."""
    text = normalized_text(value)
    words = text.split()
    if not text:
        return True
    if text in OVERLAY_WORDS or any(marker in words for marker in CLUB_MARKERS):
        return True
    if re.search(r"\b(?:TV\d*|HD|COM|ORG|NET)\b", text):
        return True
    if any(fragment in text for fragment in ("TRUC TIEP", "TRY C TIEP", "UEFA COM")):
        return True
    return False


def is_anchor(value: str, words: tuple[str, ...]) -> bool:
    text = normalized_text(value)
    return text in words


def is_formation(value: str) -> bool:
    return bool(re.fullmatch(r"\d(?:\s*[-–]\s*\d){2,4}", value.strip()))


def is_player_name(value: str) -> bool:
    candidate = clean_player_name(value)
    text = normalized_text(candidate)
    if not text or text in NON_PLAYER_WORDS or is_likely_metadata(candidate) or is_formation(candidate):
        return False
    if any(word in text for word in SUBSTITUTE_WORDS + COACH_WORDS + COMMENTATOR_WORDS):
        return False
    if re.fullmatch(r"\d+", text) or re.search(r"\d{1,2}:\d{2}", candidate):
        return False
    letters = re.sub(r"[^A-Z]", "", text)
    words = text.split()
    return len(letters) >= 3 and len(words) <= 5


def names_compatible(left: str, right: str) -> bool:
    a, b = name_key(left), name_key(right)
    if not a or not b:
        return False
    if a == b:
        return True
    if min(len(a), len(b)) >= 4 and (
        a.endswith(b) or b.endswith(a) or a.startswith(b) or b.startswith(a)
    ):
        return True
    left_words = normalized_text(left).split()
    right_words = normalized_text(right).split()
    if len(left_words) == len(right_words) and len(left_words) >= 2:
        if all(x == y or (min(len(x), len(y)) == 1 and (x.startswith(y) or y.startswith(x)))
               for x, y in zip(left_words, right_words)):
            return True
    return min(len(a), len(b)) >= 7 and SequenceMatcher(None, a, b).ratio() >= .92
