from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher


SUBSTITUTE_WORDS = ("SUBSTITUTE", "SUBSTITUTES", "SUBS", "BENCH", "RESERVES")
COACH_WORDS = ("HEAD COACH", "COACH", "MANAGER", "ENTRAINEUR", "TREINER")
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


def is_anchor(value: str, words: tuple[str, ...]) -> bool:
    text = normalized_text(value)
    return text in words


def is_formation(value: str) -> bool:
    return bool(re.fullmatch(r"\d(?:\s*[-–]\s*\d){2,4}", value.strip()))


def is_player_name(value: str) -> bool:
    _, candidate = split_inline_player(value)
    text = normalized_text(candidate)
    if not text or text in NON_PLAYER_WORDS or is_formation(candidate):
        return False
    if any(word in text for word in SUBSTITUTE_WORDS + COACH_WORDS):
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
