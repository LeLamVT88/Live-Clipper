from __future__ import annotations

import time
from pathlib import Path
import cv2

from line_up.config import PipelineConfig
from line_up.schema import DetectionResult, FrameSampleResult, LineupInterval
from line_up.scene_detector import detect_scenes
from line_up.sampler import get_scene_sample_timestamps
from line_up.ocr import get_ocr_model, extract_frames_at_timestamps, run_ocr_batch
from line_up.scorer import compute_lineup_score
from line_up.interval_proposer import propose_lineups
from line_up.multimodal import (
    compute_whole_frame_stationarity, compute_box_scaled_layout_score,
    compute_combined_multimodal_score,
)


def detect_lineups(
    video_path: str | Path,
    config: PipelineConfig | None = None,
) -> DetectionResult:
    """Detect starting lineups from broadcast football video using the validated pipeline.

    1. PySceneDetect (Adaptive + Content detectors)
    2. Scene-Adaptive Sampling (Tier 1/2/3 + micro pooling)
    3. PP-OCRv6 Tiny batch inference on CPU
    4. Discriminative lineup scoring (filtering clocks, scoreboards, non-lineup terms)
    5. CV filtering of weak evidence, with strong semantic detections protected.
    6. Content-aware interval merging and targeted OCR recovery.
    """
    vpath = Path(video_path)
    if not vpath.exists():
        raise FileNotFoundError(f"Video file does not exist: {vpath}")

    cfg = config or PipelineConfig()
    t_start = time.perf_counter()

    # Determine duration
    cap = cv2.VideoCapture(str(vpath))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_duration = total_frames / fps
    if cfg.max_scan_seconds is not None and cfg.max_scan_seconds <= 0:
        raise ValueError("max_scan_seconds must be positive or None.")
    video_duration = (
        source_duration
        if cfg.max_scan_seconds is None
        else min(source_duration, cfg.max_scan_seconds)
    )
    cap.release()

    # 1. Fast Scene Detection
    t_psd_0 = time.perf_counter()
    scenes = detect_scenes(vpath, max_duration=video_duration, config=cfg)
    time_psd = time.perf_counter() - t_psd_0

    # 2. Scene-Adaptive Sampling
    sample_timestamps = get_scene_sample_timestamps(scenes, max_duration=video_duration, config=cfg)

    # Score base samples, then verify isolated strong detections with real OCR.
    ocr_model = get_ocr_model(det_model=cfg.det_model, rec_model=cfg.rec_model)
    frame_results: list[FrameSampleResult] = []
    time_ocr = 0.0
    paired_cap = cv2.VideoCapture(str(vpath)) if cfg.use_multimodal else None

    def evaluate(timestamps: list[float]) -> None:
        nonlocal time_ocr
        valid_ts, frames = extract_frames_at_timestamps(vpath, timestamps)
        t_ocr = time.perf_counter()
        results = run_ocr_batch(frames, ocr_model)
        time_ocr += time.perf_counter() - t_ocr
        if len(results) != len(valid_ts):
            raise RuntimeError("OCR result count does not match extracted frames")
        for ts, frame, (texts, confs, boxes) in zip(valid_ts, frames, results):
            semantic, details = compute_lineup_score(texts, confs, len(boxes), config=cfg)
            static = layout = None
            score = semantic
            if paired_cap is not None:
                pair_t = ts + cfg.paired_frame_dt
                if pair_t >= video_duration:
                    pair_t = max(0.0, ts - cfg.paired_frame_dt)
                paired_cap.set(cv2.CAP_PROP_POS_MSEC, pair_t * 1000.0)
                ok, paired = paired_cap.read()
                if ok and paired is not None:
                    static = compute_whole_frame_stationarity(frame, paired)
                layout = compute_box_scaled_layout_score(boxes, frame.shape)
                score = compute_combined_multimodal_score(
                    semantic, static, layout,
                    strong_evidence=details.get("strong_evidence", False),
                )
            frame_results.append(FrameSampleResult(
                timestamp=ts, score=score, box_count=len(boxes),
                candidate_names=details.get("sample_names", []),
                jersey_numbers=details.get("sample_numbers", []),
                has_formation=details.get("formation", False),
                has_keyword=details.get("keyword", False),
                strong_evidence=details.get("strong_evidence", False),
                suppressed=details.get("suppressed", False),
                suppress_reason=details.get("reason", ""),
                content_barrier=details.get("content_barrier", False),
                ocr_score=semantic, stationarity=static, layout_score=layout,
            ))

    def propose() -> list[LineupInterval]:
        ordered = sorted(frame_results, key=lambda f: f.timestamp)
        return propose_lineups(
            [f.timestamp for f in ordered], [f.score for f in ordered], scenes,
            strong_flags=[f.strong_evidence for f in ordered],
            content_barriers=[f.content_barrier for f in ordered], config=cfg,
        )

    try:
        evaluate(sample_timestamps)
        lineups = propose()
        recovery = get_recovery_timestamps(frame_results, lineups, scenes, cfg)
        if recovery:
            evaluate(recovery)
            lineups = propose()
    finally:
        if paired_cap is not None:
            paired_cap.release()
    frame_results.sort(key=lambda f: f.timestamp)
    valid_ts = [f.timestamp for f in frame_results]

    total_time = time.perf_counter() - t_start

    stats = {
        "time_psd_seconds": round(time_psd, 2),
        "recovery_frames_requested": len(recovery),
        "time_ocr_seconds": round(time_ocr, 2),
        "total_time_seconds": round(total_time, 2),
        "ocr_fps": round(len(valid_ts) / max(0.001, time_ocr), 1),
        "sampling_reduction_ratio": round((total_frames - len(valid_ts)) / max(1, total_frames), 4),
    }

    return DetectionResult(
        video_path=vpath,
        video_duration_scanned=video_duration,
        scenes=scenes,
        sampled_frames=frame_results,
        lineups=lineups,
        processing_time_seconds=total_time,
        stats=stats,
    )


def get_recovery_timestamps(
    frames: list[FrameSampleResult],
    lineups: list[LineupInterval],
    scenes: list[tuple[float, float]],
    config: PipelineConfig,
) -> list[float]:
    """Verify uncovered structural positives; never lower the persistence threshold."""
    candidates = []
    anchors = 0
    for frame in sorted(frames, key=lambda f: (-f.score, f.timestamp)):
        if anchors >= config.max_recovery_anchors:
            break
        if (frame.score < config.strong_score_threshold or not frame.strong_evidence
                or not (frame.has_keyword or frame.has_formation)):
            continue
        t = frame.timestamp
        if any(p.start_seconds <= t <= p.end_seconds for p in lineups):
            continue
        scene = next(((s, e) for s, e in scenes if s <= t < e), None)
        if scene is None:
            continue
        neighbours = [round(t + offset, 2) for offset in
                      (-config.recovery_offset_sec, config.recovery_offset_sec)
                      if scene[0] < t + offset < scene[1]
                      and all(abs(f.timestamp - (t + offset)) > 0.5 for f in frames)
                      and all(abs(x - (t + offset)) > 0.5 for x in candidates)]
        if neighbours:
            candidates.extend(neighbours)
            anchors += 1
    return sorted(candidates)
