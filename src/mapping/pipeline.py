from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np

from mapping.consensus import resolve_frames
from mapping.frame_selector import (
    centered_candidates, select_canonical_analysis, select_top_frames, stable_groups,
)
from mapping.graphics import consolidate_graphics
from mapping.layout import analyze_layout
from mapping.number_reader import recover_frame_numbers, recover_unresolved_from_frames
from mapping.schema import FrameAnalysis
from mapping.tracking import supplement_scout_names
from mapping.validation import extraction_status
from ocr_engine import get_ocr_model, run_ocr_batch


@dataclass(frozen=True, slots=True)
class MappingConfig:
    scout_fps: float = 2.0
    fallback_scout_fps: float = 5.0
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
        if not 0 < self.scout_fps <= 10 or not 0 < self.fallback_scout_fps <= 10:
            raise ValueError("Scout frame rates must be in (0, 10]")
        if self.initial_frames < 1 or self.fallback_frames < self.initial_frames:
            raise ValueError("Expected 1 <= initial_frames <= fallback_frames")
        if self.resize_width < 320 or self.batch_size < 1 or self.min_stable_frames < 2:
            raise ValueError("Invalid mapping configuration")


@dataclass(slots=True)
class _Sample:
    timestamp: float
    frame: np.ndarray
    analysis: FrameAnalysis | None = None


def _sample_video(
    video_path: Path, start: float, end: float, config: MappingConfig, *, fps: float | None = None,
) -> list[_Sample]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
    if not np.isfinite([start, end]).all() or not 0 <= start < end:
        raise ValueError("Expected a finite interval with 0 <= start < end")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    samples: list[_Sample] = []
    try:
        previous_thumbnail: np.ndarray | None = None
        previous_actual = -1.0
        for target in np.arange(start, end, 1.0 / (fps or config.scout_fps)):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(target) * 1000)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            height = round(frame.shape[0] * config.resize_width / frame.shape[1])
            frame = cv2.resize(frame, (config.resize_width, height), interpolation=cv2.INTER_AREA)
            actual = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 or float(target)
            thumbnail = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 18))
            if (previous_thumbnail is not None and abs(actual - previous_actual) < 1e-4
                    and np.array_equal(thumbnail, previous_thumbnail)):
                continue
            samples.append(_Sample(actual, frame))
            previous_thumbnail, previous_actual = thumbnail, actual
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
            sample.analysis = analyze_layout(
                texts, scores, boxes, sample.frame.shape, sample.timestamp, sharpness,
            )
    return time.perf_counter() - began


def _reanalyze(
    selected: list[_Sample], model: object, config: MappingConfig,
) -> tuple[list[FrameAnalysis], float, list[dict[str, object]]]:
    began = time.perf_counter()
    fresh = [_Sample(sample.timestamp, sample.frame) for sample in selected]
    _analyse_samples(fresh, model, config)
    for detailed, scout in zip(fresh, selected):
        if detailed.analysis is not None:
            supplement_scout_names(detailed.analysis, scout.analysis)
    local_evidence = [evidence for sample in fresh if sample.analysis is not None
                      for evidence in recover_frame_numbers(sample.frame, sample.analysis, model)]
    analyses = [sample.analysis for sample in fresh if sample.analysis is not None]
    return analyses, time.perf_counter() - began, local_evidence


def _resolve_group(
    group: list[_Sample], model: object, config: MappingConfig, index: int,
) -> tuple[dict[str, object], float]:
    selected = select_top_frames(group, config.initial_frames)
    analyses, extraction_seconds, local_evidence = _reanalyze(selected, model, config)
    players, issues = resolve_frames(analyses)
    if issues and len(group) > len(selected):
        selected = select_top_frames(group, config.fallback_frames)
        analyses, extra_seconds, local_evidence = _reanalyze(selected, model, config)
        extraction_seconds += extra_seconds
        players, issues = resolve_frames(analyses)

    canonical = select_canonical_analysis(analyses)
    unresolved = [player for player in players if not player["number_confirmed"]]
    unselected = [sample for sample in group
                  if all(sample is not chosen for chosen in selected)]
    if unresolved and unselected:
        neighbours = sorted(unselected, key=lambda sample: abs(sample.timestamp - canonical.timestamp))[:3]
        rescue_analyses, rescue_evidence = recover_unresolved_from_frames(
            [(sample.timestamp, sample.frame) for sample in neighbours], players, model,
        )
        analyses.extend(rescue_analyses)
        local_evidence.extend(rescue_evidence)
        selected = sorted([*selected, *neighbours], key=lambda sample: sample.timestamp)
        players, issues = resolve_frames(analyses)

    canonical = select_canonical_analysis([analysis for analysis in analyses if analysis.raw_ocr])
    roles = sorted({panel.role for analysis in analyses for panel in analysis.panels
                    if panel.role.startswith("starter_")})
    complete = len(players) == 11 and not issues and all(player["confirmed"] for player in players)
    return {
        "graphic_index": index,
        "start_seconds": round(group[0].timestamp, 3),
        "end_seconds": round(group[-1].timestamp, 3),
        "best_frame_timestamp": round(canonical.timestamp, 3),
        "selected_frame_timestamps": [round(sample.timestamp, 3) for sample in selected],
        "selection_tier": len(selected),
        "panel_roles": roles,
        "layout": "+".join(role.removeprefix("starter_") for role in roles) or "unknown",
        "status": "complete" if complete else "partial",
        "issues": issues,
        "players": players,
        "local_number_evidence": local_evidence,
        "ocr_evidence": [analysis.raw_ocr | {"timestamp": round(analysis.timestamp, 3)}
                         for analysis in analyses],
    }, extraction_seconds


def _find_groups(
    path: Path, start: float, end: float, config: MappingConfig, scout_model: object,
) -> tuple[list[list[_Sample]], float, int, int, bool]:
    samples = _sample_video(path, start, end, config)
    seconds = _analyse_samples(samples, scout_model, config)
    groups = stable_groups(samples, fps=config.scout_fps, min_frames=config.min_stable_frames)
    initial_count = len(samples)
    if groups or config.fallback_scout_fps <= config.scout_fps:
        return groups, seconds, initial_count, 0, False

    fallback = _sample_video(path, start, end, config, fps=config.fallback_scout_fps)
    seconds += _analyse_samples(fallback, scout_model, config)
    groups = stable_groups(
        fallback, fps=config.fallback_scout_fps,
        min_frames=config.min_stable_frames, allow_weak=True,
    )
    return groups, seconds, initial_count, len(fallback), True


def extract_lineup_graphics(
    video_path: str | Path, start: float, end: float, config: MappingConfig | None = None,
    *, scout_model: object | None = None, extraction_model: object | None = None,
) -> dict[str, object]:
    """Select top-K lineup frames, map player cards, and retain unresolved evidence."""
    cfg, path, began = config or MappingConfig(), Path(video_path), time.perf_counter()
    scout_model = scout_model or get_ocr_model(cfg.scout_det_model, cfg.scout_rec_model)
    groups, scout_seconds, initial_count, fallback_count, used_scout_fallback = _find_groups(
        path, start, end, cfg, scout_model,
    )

    graphics: list[dict[str, object]] = []
    extraction_seconds = 0.0
    if groups:
        extraction_model = extraction_model or get_ocr_model(
            cfg.extraction_det_model, cfg.extraction_rec_model,
        )
        for index, group in enumerate(groups, 1):
            graphic, seconds = _resolve_group(group, extraction_model, cfg, index)
            graphic["scout_fallback_used"] = used_scout_fallback
            graphics.append(graphic)
            extraction_seconds += seconds
        graphics = consolidate_graphics(graphics)

    return {
        "video_path": str(path),
        "interval": {"start_seconds": start, "end_seconds": end},
        "graphics": graphics,
        "stats": {
            "status": extraction_status(graphics),
            "scout_fps": cfg.fallback_scout_fps if used_scout_fallback else cfg.scout_fps,
            "initial_scout_frames": initial_count,
            "fallback_scout_frames": fallback_count,
            "scout_frames": initial_count + fallback_count,
            "scout_fallback_used": used_scout_fallback,
            "stable_graphics": len(groups),
            "resolved_graphics": len(graphics),
            "scout_ocr_seconds": round(scout_seconds, 3),
            "extraction_ocr_seconds": round(extraction_seconds, 3),
            "total_seconds": round(time.perf_counter() - began, 3),
        },
    }
