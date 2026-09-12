from __future__ import annotations

from dataclasses import dataclass, field


BBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class ExclusionRegion:
    role: str
    box: BBox
    reason: str


@dataclass(frozen=True, slots=True)
class OCRToken:
    index: int
    text: str
    confidence: float
    box: BBox

    @property
    def center(self) -> tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


@dataclass(slots=True)
class Panel:
    role: str
    box: BBox
    name_tokens: list[OCRToken]
    token_indexes: set[int] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class PlayerCard:
    """One visual lineup slot with independent name and number regions."""

    panel_role: str
    name: str
    card_box: BBox
    name_box: BBox
    number_box: BBox
    name_confidence: float
    jersey_number: int | None = None


@dataclass(slots=True)
class PlayerObservation:
    timestamp: float
    panel_role: str
    name: str
    name_confidence: float
    name_box: BBox
    jersey_number: int | None = None
    number_confidence: float = 0.0
    number_box: BBox | None = None
    number_source: str = "unresolved"
    pair_confidence: float = 0.0
    number_candidates: list[int] = field(default_factory=list)
    # Scores are local to one timestamp. OCR variants improve this score but do
    # not become independent temporal votes.
    number_candidate_scores: dict[int, float] = field(default_factory=dict)

    def to_evidence(self) -> dict[str, object]:
        return {
            "timestamp": round(self.timestamp, 3),
            "panel_role": self.panel_role,
            "name": self.name,
            "name_confidence": round(self.name_confidence, 4),
            "name_box": [round(v, 5) for v in self.name_box],
            "jersey_number": self.jersey_number,
            "number_candidates": self.number_candidates,
            "number_candidate_scores": {
                str(number): round(score, 4)
                for number, score in sorted(self.number_candidate_scores.items())
            },
            "number_confidence": round(self.number_confidence, 4),
            "number_box": None if self.number_box is None else [round(v, 5) for v in self.number_box],
            "number_source": self.number_source,
            "pair_confidence": round(self.pair_confidence, 4),
        }


@dataclass(slots=True)
class FrameAnalysis:
    timestamp: float
    panels: list[Panel]
    observations: list[PlayerObservation]
    sharpness: float
    score: float
    issues: list[str]
    raw_ocr: dict[str, object]
    cards: list[PlayerCard] = field(default_factory=list)
    exclusions: list[ExclusionRegion] = field(default_factory=list)

    @property
    def starter_observations(self) -> list[PlayerObservation]:
        return [o for o in self.observations if o.panel_role.startswith("starter_")]
