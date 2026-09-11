from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np

from mapping.consensus import resolve_frames
from mapping.layout import analyze_layout
from mapping.number_reader import recover_frame_numbers
from mapping.schema import FrameAnalysis
from mapping.text import name_key, names_compatible
from ocr_engine import get_ocr_model, run_ocr_batch


@dataclass(frozen=True, slots=True)
class MappingConfig:
    scout_fps: float = 2.0
    resize_width: int = 1280
    batch_size: int = 8
    initial_frames: int = 3
    fallback_frames: int = 7
    min_stable_frames: int = 2
    scout_det_model: str = "PP-OCRv6_tiny_det"
    scout_rec_model: str = "PP-OCRv6_tiny_rec"
    extraction_det_model: str = "PP-OCRv6_small_det"
    extraction_rec_model: str = "PP-OCRv6_small_rec"

    def __post_init__(self) -> None:
        if not 0 < self.scout_fps <= 10:
            raise ValueError("scout_fps must be in (0, 10]")
        if self.initial_frames < 1 or self.fallback_frames < self.initial_frames:
            raise ValueError("Expected 1 <= initial_frames <= fallback_frames")
        if self.resize_width < 320 or self.batch_size < 1 or self.min_stable_frames < 2:
            raise ValueError("Invalid mapping configuration")


@dataclass(slots=True)
class _Sample:
    timestamp: float
    frame: np.ndarray
    analysis: FrameAnalysis | None = None


def _sample_video(video_path: Path, start: float, end: float, config: MappingConfig) -> list[_Sample]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
    if not np.isfinite([start, end]).all() or not 0 <= start < end:
        raise ValueError("Expected a finite interval with 0 <= start < end")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    samples = []
    try:
        for target in np.arange(start, end, 1.0 / config.scout_fps):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(target) * 1000)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            height = round(frame.shape[0] * config.resize_width / frame.shape[1])
            frame = cv2.resize(frame, (config.resize_width, height), interpolation=cv2.INTER_AREA)
            actual = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            samples.append(_Sample(actual if actual > 0 else float(target), frame))
    finally:
        cap.release()
    return samples


def _analyse_samples(samples: list[_Sample], model: object, config: MappingConfig) -> float:
    began = time.perf_counter()
    for offset in range(0, len(samples), config.batch_size):
        batch = samples[offset:offset + config.batch_size]
        outputs = run_ocr_batch([sample.frame for sample in batch], model)
        if len(outputs) != len(batch):
            raise RuntimeError("OCR result count does not match frame count")
        for sample, (texts, scores, boxes) in zip(batch, outputs):
            gray = cv2.cvtColor(sample.frame, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
            sample.analysis = analyze_layout(texts, scores, boxes, sample.frame.shape, sample.timestamp, sharpness)
    return time.perf_counter() - began


def _unique_names(analysis: FrameAnalysis) -> list[str]:
    result = []
    for observation in analysis.starter_observations:
        if not any(names_compatible(observation.name, name) for name in result):
            result.append(observation.name)
    return result


def _similarity(left: FrameAnalysis, right: FrameAnalysis) -> float:
    a, b = _unique_names(left), _unique_names(right)
    if not a or not b:
        return 0.0
    matches = sum(any(names_compatible(name, other) for other in b) for name in a)
    return matches / max(len(a), len(b))


def _candidate(sample: _Sample) -> bool:
    if sample.analysis is None:
        return False
    count = len(_unique_names(sample.analysis))
    return 7 <= count <= 13 and any(panel.role.startswith("starter_") for panel in sample.analysis.panels)


def _stable_groups(samples: list[_Sample], config: MappingConfig) -> list[list[_Sample]]:
    groups: list[list[_Sample]] = []
    current: list[_Sample] = []
    max_gap = 1.6 / config.scout_fps
    for sample in samples:
        if not _candidate(sample):
            if len(current) >= config.min_stable_frames:
                groups.append(current)
            current = []
            continue
        if current and (sample.timestamp - current[-1].timestamp > max_gap
                        or _similarity(current[-1].analysis, sample.analysis) < .55):
            if len(current) >= config.min_stable_frames:
                groups.append(current)
            current = []
        current.append(sample)
    if len(current) >= config.min_stable_frames:
        groups.append(current)

    merged: list[list[_Sample]] = []
    for group in groups:
        if (merged and group[0].timestamp - merged[-1][-1].timestamp <= 2.0
                and _similarity(merged[-1][-1].analysis, group[0].analysis) >= .70):
            merged[-1].extend(group)
        else:
            merged.append(group)
    return merged


def centered_candidates(group: list[_Sample], best_index: int, count: int) -> list[_Sample]:
    """Take candidate frames around the best one, never unrelated raw neighbours."""
    count = min(count, len(group))
    start = max(0, min(best_index - count // 2, len(group) - count))
    return group[start:start + count]


def _reanalyze(selected: list[_Sample], model: object, config: MappingConfig) -> tuple[list[FrameAnalysis], float, list[dict[str, object]]]:
    began = time.perf_counter()
    fresh = [_Sample(sample.timestamp, sample.frame) for sample in selected]
    _analyse_samples(fresh, model, config)
    local_evidence = []
    for sample in fresh:
        local_evidence.extend(recover_frame_numbers(sample.frame, sample.analysis, model))
    return ([sample.analysis for sample in fresh if sample.analysis is not None],
            time.perf_counter() - began, local_evidence)


def _resolve_group(group: list[_Sample], extraction_model: object, config: MappingConfig, index: int) -> tuple[dict[str, object], float]:
    best_index = max(range(len(group)), key=lambda i: group[i].analysis.score)
    selected = centered_candidates(group, best_index, config.initial_frames)
    analyses, extraction_seconds, local_evidence = _reanalyze(selected, extraction_model, config)
    players, issues = resolve_frames(analyses)
    used_fallback = False
    if issues and len(group) > len(selected):
        selected = centered_candidates(group, best_index, config.fallback_frames)
        analyses, extra, local_evidence = _reanalyze(selected, extraction_model, config)
        extraction_seconds += extra
        players, issues = resolve_frames(analyses)
        used_fallback = True

    roles = sorted({panel.role for analysis in analyses for panel in analysis.panels
                    if panel.role.startswith("starter_")})
    complete = len(players) == 11 and not issues and all(player["confirmed"] for player in players)
    return {
        "graphic_index": index,
        "start_seconds": round(group[0].timestamp, 3),
        "end_seconds": round(group[-1].timestamp, 3),
        "best_frame_timestamp": round(group[best_index].timestamp, 3),
        "selected_frame_timestamps": [round(sample.timestamp, 3) for sample in selected],
        "selection_tier": 7 if used_fallback else 3,
        "panel_roles": roles,
        "layout": "+".join(role.removeprefix("starter_") for role in roles) or "unknown",
        "status": "complete" if complete else "partial",
        "issues": issues,
        "players": players,
        "local_number_evidence": local_evidence,
        "ocr_evidence": [analysis.raw_ocr | {"timestamp": round(analysis.timestamp, 3)} for analysis in analyses],
    }, extraction_seconds


def _graphic_name_overlap(left: dict[str, object], right: dict[str, object]) -> int:
    """Count one-to-one compatible names across two phases of a lineup animation."""
    unmatched = list(right["players"])
    matches = 0
    for player in left["players"]:
        match = next((candidate for candidate in unmatched
                      if names_compatible(player["name"], candidate["name"])), None)
        if match is not None:
            unmatched.remove(match)
            matches += 1
    return matches


def _graphic_quality(graphic: dict[str, object]) -> tuple[object, ...]:
    players = graphic["players"]
    return (
        graphic["status"] == "complete",
        -abs(len(players) - 11),
        sum(bool(player["confirmed"]) for player in players),
        "list" in graphic["layout"],
        len(players),
    )


def _merge_player_evidence(target: dict[str, object], source: dict[str, object]) -> None:
    """Fill missing facts only; never overwrite a confirmed pair with a conflict."""
    if names_compatible(target["name"], source["name"]):
        target_key = name_key(target["name"])
        source_key = name_key(source["name"])
        if len(source_key) > len(target_key):
            target["name"] = source["name"]
    if target["shirt_number"] is None and source["number_confirmed"]:
        target["shirt_number"] = source["shirt_number"]
        target["number_confirmed"] = True
        target["number_status"] = source["number_status"]
        target["number_source"] = source["number_source"]
        target["pair_confidence"] = source["pair_confidence"]
    target["confirmed"] = bool(target["name_confirmed"] and target["number_confirmed"])
    target["number_candidates"] = sorted(set(target["number_candidates"]) | set(source["number_candidates"]))
    target["evidence_timestamps"] = sorted(
        set(target["evidence_timestamps"]) | set(source["evidence_timestamps"])
    )
    target["observations"].extend(source["observations"])


def _refresh_graphic(graphic: dict[str, object]) -> None:
    players = graphic["players"]
    duplicate_names = len({name_key(player["name"]) for player in players}) != len(players)
    confirmed_numbers = [player["shirt_number"] for player in players if player["number_confirmed"]]
    duplicate_numbers = len(set(confirmed_numbers)) != len(confirmed_numbers)
    issues = []
    if len(players) != 11:
        issues.append(f"resolved_player_count_{len(players)}")
    if duplicate_names:
        issues.append("duplicate_names")
    if duplicate_numbers:
        issues.append("duplicate_shirt_numbers")
    if any(not player["name_confirmed"] for player in players):
        issues.append("unconfirmed_names")
    if any(not player["number_confirmed"] for player in players):
        issues.append("unconfirmed_numbers")
    graphic["issues"] = issues
    graphic["status"] = "complete" if len(players) == 11 and not issues else "partial"
    for slot, player in enumerate(players, 1):
        player["slot_index"] = slot


def _merge_graphic_cluster(cluster: list[dict[str, object]]) -> dict[str, object]:
    primary = max(cluster, key=_graphic_quality)
    merged = primary.copy()
    merged["players"] = [player.copy() | {
        "number_candidates": list(player["number_candidates"]),
        "evidence_timestamps": list(player["evidence_timestamps"]),
        "observations": list(player["observations"]),
    } for player in primary["players"]]

    for phase in cluster:
        if phase is primary:
            continue
        for source in phase["players"]:
            target = next((player for player in merged["players"]
                           if names_compatible(player["name"], source["name"])), None)
            if target is None and source["number_confirmed"]:
                same_number = [player for player in merged["players"]
                               if player["number_confirmed"]
                               and player["shirt_number"] == source["shirt_number"]]
                target = same_number[0] if len(same_number) == 1 else None
            if target is not None:
                _merge_player_evidence(target, source)
            elif len(merged["players"]) < 11:
                merged["players"].append(source.copy() | {
                    "number_candidates": list(source["number_candidates"]),
                    "evidence_timestamps": list(source["evidence_timestamps"]),
                    "observations": list(source["observations"]),
                })

    merged["start_seconds"] = min(phase["start_seconds"] for phase in cluster)
    merged["end_seconds"] = max(phase["end_seconds"] for phase in cluster)
    merged["selected_frame_timestamps"] = sorted({timestamp for phase in cluster
                                                  for timestamp in phase["selected_frame_timestamps"]})
    merged["selection_tier"] = max(phase["selection_tier"] for phase in cluster)
    roles = sorted({role for phase in cluster for role in phase["panel_roles"]})
    merged["panel_roles"] = roles
    merged["layout"] = "+".join(role.removeprefix("starter_") for role in roles) or "unknown"
    merged["local_number_evidence"] = [item for phase in cluster
                                       for item in phase["local_number_evidence"]]
    merged["ocr_evidence"] = [item for phase in cluster for item in phase["ocr_evidence"]]
    _refresh_graphic(merged)
    return merged


def consolidate_graphics(graphics: list[dict[str, object]]) -> list[dict[str, object]]:
    """Join list/pitch phases that show the same starting eleven."""
    clusters: list[list[dict[str, object]]] = []
    for graphic in graphics:
        cluster = next((items for items in reversed(clusters)
                        if graphic["start_seconds"] - items[-1]["end_seconds"] <= 3.0
                        and _graphic_name_overlap(items[-1], graphic) >= 6), None)
        if cluster is None:
            clusters.append([graphic])
        else:
            cluster.append(graphic)
    result = [_merge_graphic_cluster(cluster) for cluster in clusters]
    for index, graphic in enumerate(result, 1):
        graphic["graphic_index"] = index
    return result


def extract_lineup_graphics(
    video_path: str | Path, start: float, end: float, config: MappingConfig | None = None,
    *, scout_model: object | None = None, extraction_model: object | None = None,
) -> dict[str, object]:
    """Find stable starter panels, map names to numbers, and retain unresolved evidence."""
    cfg = config or MappingConfig()
    path = Path(video_path)
    began = time.perf_counter()
    samples = _sample_video(path, start, end, cfg)
    scout_model = scout_model or get_ocr_model(cfg.scout_det_model, cfg.scout_rec_model)
    scout_seconds = _analyse_samples(samples, scout_model, cfg)
    groups = _stable_groups(samples, cfg)
    graphics = []
    extraction_seconds = 0.0
    if groups:
        extraction_model = extraction_model or get_ocr_model(cfg.extraction_det_model, cfg.extraction_rec_model)
        for index, group in enumerate(groups, 1):
            graphic, seconds = _resolve_group(group, extraction_model, cfg, index)
            graphics.append(graphic)
            extraction_seconds += seconds
        graphics = consolidate_graphics(graphics)
    return {
        "video_path": str(path),
        "interval": {"start_seconds": start, "end_seconds": end},
        "graphics": graphics,
        "stats": {
            "status": "complete" if any(g["status"] == "complete" for g in graphics)
                      else ("partial" if graphics else "no_stable_graphic"),
            "scout_fps": cfg.scout_fps,
            "scout_frames": len(samples),
            "stable_graphics": len(groups),
            "resolved_graphics": len(graphics),
            "scout_ocr_seconds": round(scout_seconds, 3),
            "extraction_ocr_seconds": round(extraction_seconds, 3),
            "total_seconds": round(time.perf_counter() - began, 3),
        },
    }
