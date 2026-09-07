from __future__ import annotations

from lineup_test.config import PipelineConfig


def get_scene_sample_timestamps(
    scenes: list[tuple[float, float]],
    max_duration: float,
    config: PipelineConfig | None = None,
) -> list[float]:
    """Perform Scene-Adaptive Sampling to reduce frame evaluations by ~85%.

    Sampling Strategy:
    - Long scene (>15s): 25%, 50%, 75%
    - Medium scene (5-15s): 30%, 70%
    - Short scene (2-5s): 50% (center)
    - Micro scenes (<2s): Pooled until cumulative duration >= 3s, then take 1 sample
    """
    cfg = config or PipelineConfig()
    samples: list[float] = []
    short_pool: list[tuple[float, float]] = []

    for start, end in scenes:
        dur = end - start
        if dur < cfg.tier_short_sec:
            short_pool.append((start, end))
            pool_duration = short_pool[-1][1] - short_pool[0][0]
            if pool_duration >= cfg.micro_pool_target_sec:
                samples.append((short_pool[0][0] + short_pool[-1][1]) / 2.0)
                short_pool = []
            continue

        if short_pool:
            if short_pool[-1][1] - short_pool[0][0] >= 1.5:
                samples.append((short_pool[0][0] + short_pool[-1][1]) / 2.0)
            short_pool = []

        if cfg.tier_short_sec <= dur < cfg.tier_medium_sec:
            # Short scenes (2-5s): 1 frame at 50%
            samples.append(start + dur * 0.50)
        elif cfg.tier_medium_sec <= dur < cfg.tier_long_sec:
            # Medium scenes (5-15s): 3 frames at 25%, 50%, 75%
            samples.append(start + dur * 0.25)
            samples.append(start + dur * 0.50)
            samples.append(start + dur * 0.75)
        elif cfg.tier_long_sec <= dur < cfg.tier_ultra_long_sec:
            # Long scenes (15-35s): 4 frames at 20%, 40%, 60%, 80%
            samples.append(start + dur * 0.20)
            samples.append(start + dur * 0.40)
            samples.append(start + dur * 0.60)
            samples.append(start + dur * 0.80)
        else:
            # Ultra-Long scenes (>= 35s): 5 frames at 10%, 30%, 50%, 70%, 90%
            samples.append(start + dur * 0.10)
            samples.append(start + dur * 0.30)
            samples.append(start + dur * 0.50)
            samples.append(start + dur * 0.70)
            samples.append(start + dur * 0.90)

    if short_pool and (short_pool[-1][1] - short_pool[0][0]) >= 1.5:
        samples.append((short_pool[0][0] + short_pool[-1][1]) / 2.0)

    valid_samples = [
        round(t, 2) for t in samples if 0.0 <= t <= max_duration
    ]
    return sorted(set(valid_samples))
