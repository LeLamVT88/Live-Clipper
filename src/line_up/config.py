from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PipelineConfig:
    """Configuration parameters for the lineup detection pipeline."""

    # Video Scan Window. None scans the complete input video.
    max_scan_seconds: float | None = None

    # PySceneDetect Settings
    adaptive_threshold: float = 3.0
    content_threshold: float = 27.0
    min_scene_len_frames: int = 12
    frame_skip: int = 1
    auto_downscale: bool = True

    # Scene-Adaptive Sampling Tiers
    tier_ultra_long_sec: float = 35.0  # >= 35s: 5 frames (10%, 30%, 50%, 70%, 90%)
    tier_long_sec: float = 15.0        # 15-35s: 4 frames (20%, 40%, 60%, 80%)
    tier_medium_sec: float = 5.0       # 5-15s: 3 frames (25%, 50%, 75%)
    tier_short_sec: float = 2.0        # 2-5s: 1 frame (50%)
    micro_pool_target_sec: float = 3.0 # <2s: pooled until >= 3s, then 1 center sample
    ultra_long_min_samples: int = 7
    max_sample_gap_sec: float = 10.0

    # PP-OCRv6 Model Settings
    det_model: str = "PP-OCRv6_tiny_det"
    rec_model: str = "PP-OCRv6_tiny_rec"
    device: str = "cpu"
    min_box_confidence: float = 0.45

    # Discriminative Scoring Gates
    min_candidate_names: int = 4
    name_saturation_count: float = 8.0
    number_saturation_count: float = 6.0
    structure_bonus: float = 0.25
    decision_threshold: float = 0.45
    fallback_decision_threshold: float = 0.35
    min_unstructured_numbers: int = 2
    unstructured_score_cap: float = 0.40
    strong_name_count: int = 6
    strong_number_count: int = 4

    # Temporal Persistence & Post-Processing
    min_lineup_duration_sec: float = 8.0
    min_positive_samples: int = 2
    min_strong_samples: int = 1
    strong_score_threshold: float = 0.70
    cluster_break_score: float = 0.10
    intra_scene_cluster_gap_sec: float = 12.0
    inter_proposal_merge_gap_sec: float = 4.0
    nms_iou_threshold: float = 0.20
    max_lineup_intervals: int = 2
    contiguous_lineup_merge_gap_sec: float = 10.0  # Merges back-to-back lineups (e.g. Bundesliga)

    # CV only filters weak evidence; strong semantic lineups remain protected.
    use_multimodal: bool = True
    paired_frame_dt: float = 0.5
    recovery_offset_sec: float = 1.5
    max_recovery_anchors: int = 2
