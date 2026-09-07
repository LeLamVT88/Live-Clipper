from __future__ import annotations

import time
from pathlib import Path
import cv2

from lineup_test.config import PipelineConfig
from lineup_test.schema import DetectionResult, FrameSampleResult
from lineup_test.scene_detector import detect_scenes
from lineup_test.sampler import get_scene_sample_timestamps
from lineup_test.ocr_engine import get_ocr_model, extract_frames_at_timestamps, run_ocr_batch
from lineup_test.scorer import compute_lineup_score
from lineup_test.interval_proposer import propose_lineups


def detect_lineups(
    video_path: str | Path,
    config: PipelineConfig | None = None,
) -> DetectionResult:
    """Detect starting lineups from broadcast football video using the validated pipeline.

    1. PySceneDetect (Adaptive + Content detectors)
    2. Scene-Adaptive Sampling (Tier 1/2/3 + micro pooling)
    3. PP-OCRv6 Tiny batch inference on CPU
    4. Discriminative lineup scoring (filtering clocks, scoreboards, non-lineup terms)
    5. Temporal persistence snapping, Non-Maximum Suppression, and contiguous merging.
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
    video_duration = min(total_frames / fps, cfg.max_scan_seconds)
    cap.release()

    # 1. Fast Scene Detection
    t_psd_0 = time.perf_counter()
    scenes = detect_scenes(vpath, max_duration=video_duration, config=cfg)
    time_psd = time.perf_counter() - t_psd_0

    # 2. Scene-Adaptive Sampling
    sample_timestamps = get_scene_sample_timestamps(scenes, max_duration=video_duration, config=cfg)

    # 3. Frame Extraction
    valid_ts, raw_frames = extract_frames_at_timestamps(vpath, sample_timestamps)

    # 4. PP-OCRv6 Tiny Inference
    ocr_model = get_ocr_model(det_model=cfg.det_model, rec_model=cfg.rec_model)
    t_ocr_0 = time.perf_counter()
    ocr_results = run_ocr_batch(raw_frames, ocr_model)
    time_ocr = time.perf_counter() - t_ocr_0

    # 5. Discriminative Scoring
    frame_results: list[FrameSampleResult] = []
    numeric_scores: list[float] = []
    for ts, (texts, confs, boxes) in zip(valid_ts, ocr_results):
        score, details = compute_lineup_score(texts, confs, len(boxes), config=cfg)
        numeric_scores.append(score)
        frame_results.append(
            FrameSampleResult(
                timestamp=ts,
                score=score,
                box_count=len(boxes),
                candidate_names=details.get("sample_names", []),
                jersey_numbers=details.get("sample_numbers", []),
                has_formation=details.get("formation", False),
                has_keyword=details.get("keyword", False),
                strong_evidence=details.get("strong_evidence", False),
                suppressed=details.get("suppressed", False),
                suppress_reason=details.get("reason", ""),
            )
        )

    # 6. Proposal, Persistence, NMS, and Merging
    lineups = propose_lineups(
        valid_ts,
        numeric_scores,
        scenes,
        strong_flags=[frame.strong_evidence for frame in frame_results],
        config=cfg,
    )

    total_time = time.perf_counter() - t_start

    stats = {
        "time_psd_seconds": round(time_psd, 2),
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
