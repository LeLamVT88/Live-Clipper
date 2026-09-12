from __future__ import annotations

from typing import Protocol, TypeVar

from mapping.schema import FrameAnalysis
from mapping.text import names_compatible


class SampleLike(Protocol):
    timestamp: float
    analysis: FrameAnalysis | None


SampleT = TypeVar("SampleT", bound=SampleLike)


def unique_names(analysis: FrameAnalysis) -> list[str]:
    names = [observation.name for observation in analysis.starter_observations]
    if not names:
        names = [card.name for card in analysis.cards if card.panel_role == "weak_pitch"]
    result: list[str] = []
    for name in names:
        if not any(names_compatible(name, existing) for existing in result):
            result.append(name)
    return result


def frame_similarity(left: FrameAnalysis, right: FrameAnalysis) -> float:
    left_names, right_names = unique_names(left), unique_names(right)
    if not left_names or not right_names:
        return 0.0
    matches = sum(any(names_compatible(name, other) for other in right_names)
                  for name in left_names)
    return matches / max(len(left_names), len(right_names))


def is_candidate(sample: SampleLike, *, allow_weak: bool = False) -> bool:
    if sample.analysis is None:
        return False
    count = len(unique_names(sample.analysis))
    roles = {panel.role for panel in sample.analysis.panels}
    return 7 <= count <= 13 and (
        any(role.startswith("starter_") for role in roles)
        or (allow_weak and "weak_pitch" in roles)
    )


def stable_groups(
    samples: list[SampleT], *, fps: float, min_frames: int = 2, allow_weak: bool = False,
) -> list[list[SampleT]]:
    groups: list[list[SampleT]] = []
    current: list[SampleT] = []
    max_gap = 1.6 / fps
    for sample in samples:
        if not is_candidate(sample, allow_weak=allow_weak):
            if len(current) >= min_frames:
                groups.append(current)
            current = []
            continue
        if current and (
            sample.timestamp - current[-1].timestamp > max_gap
            or frame_similarity(current[-1].analysis, sample.analysis) < .55
        ):
            if len(current) >= min_frames:
                groups.append(current)
            current = []
        current.append(sample)
    if len(current) >= min_frames:
        groups.append(current)

    merged: list[list[SampleT]] = []
    for group in groups:
        if (merged and group[0].timestamp - merged[-1][-1].timestamp <= 2.0
                and frame_similarity(merged[-1][-1].analysis, group[0].analysis) >= .70):
            merged[-1].extend(group)
        else:
            merged.append(group)
    return merged


def analysis_quality(analysis: FrameAnalysis | None) -> tuple[object, ...]:
    if analysis is None:
        return False, -99, 0, 0.0, 0.0
    names = unique_names(analysis)
    paired = sum(observation.jersey_number is not None
                 for observation in analysis.starter_observations)
    strong = any(panel.role.startswith("starter_") for panel in analysis.panels)
    return strong, -abs(len(names) - 11), paired, analysis.score, analysis.sharpness


def _quality(sample: SampleLike) -> tuple[object, ...]:
    return analysis_quality(sample.analysis)


def select_canonical_analysis(analyses: list[FrameAnalysis]) -> FrameAnalysis:
    if not analyses:
        raise ValueError("Cannot select a canonical frame from an empty analysis list")
    return max(analyses, key=analysis_quality)


def select_top_frames(group: list[SampleT], limit: int, *, min_gap: float = .25) -> list[SampleT]:
    """Cover every visible name twice, then use quality and temporal diversity."""
    ordered = sorted(group, key=lambda sample: sample.timestamp)
    count = min(limit, len(ordered))
    if count == len(ordered):
        return ordered

    universe: list[str] = []
    sample_names: dict[int, set[int]] = {}
    frequencies: dict[int, int] = {}
    for sample in ordered:
        indexes: set[int] = set()
        for name in unique_names(sample.analysis):
            index = next((i for i, known in enumerate(universe)
                          if names_compatible(name, known)), None)
            if index is None:
                universe.append(name)
                index = len(universe) - 1
            indexes.add(index)
        sample_names[id(sample)] = indexes
        for index in indexes:
            frequencies[index] = frequencies.get(index, 0) + 1

    if not universe:
        return sorted(ordered, key=_quality, reverse=True)[:count]

    coverage = {index: 0 for index in range(len(universe))}
    selected: list[SampleT] = []
    remaining = list(ordered)
    while remaining and len(selected) < count:
        def selection_key(sample: SampleT) -> tuple[object, ...]:
            gain = sum(max(0, min(2, frequencies[index]) - coverage[index])
                       for index in sample_names[id(sample)])
            distance = min((abs(sample.timestamp - item.timestamp) for item in selected), default=0.0)
            return gain, distance, _quality(sample)

        winner = max(remaining, key=selection_key)
        selected.append(winner)
        remaining = [sample for sample in remaining if sample is not winner]
        for index in sample_names[id(winner)]:
            coverage[index] += 1
    return sorted(selected, key=lambda sample: sample.timestamp)


def centered_candidates(group: list[SampleT], best_index: int, count: int) -> list[SampleT]:
    """Backward-compatible helper retained for callers and tests."""
    count = min(count, len(group))
    start = max(0, min(best_index - count // 2, len(group) - count))
    return group[start:start + count]
