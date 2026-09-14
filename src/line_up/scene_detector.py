from __future__ import annotations

from pathlib import Path
from scenedetect import SceneManager, open_video
from scenedetect.detectors import AdaptiveDetector, ContentDetector

from line_up.config import PipelineConfig


def detect_scenes(
    video_path: Path,
    max_duration: float,
    config: PipelineConfig | None = None,
) -> list[tuple[float, float]]:
    """Detect broadcast scene boundaries using combined Adaptive + Content detectors.

    AdaptiveDetector catches gradual graphic transitions, wipes, and lower-thirds.
    ContentDetector snaps hard physical camera cuts cleanly.
    """
    cfg = config or PipelineConfig()
    video = open_video(str(video_path))
    manager = SceneManager()
    manager.auto_downscale = cfg.auto_downscale

    manager.add_detector(
        AdaptiveDetector(
            adaptive_threshold=cfg.adaptive_threshold,
            min_scene_len=cfg.min_scene_len_frames,
            window_width=2,
        )
    )
    manager.add_detector(
        ContentDetector(
            threshold=cfg.content_threshold,
            min_scene_len=cfg.min_scene_len_frames,
        )
    )

    manager.detect_scenes(
        video=video,
        end_time=max_duration,
        frame_skip=cfg.frame_skip,
        show_progress=False,
    )

    raw_scenes = manager.get_scene_list(start_in_scene=True)
    scenes: list[tuple[float, float]] = []
    for start_time, end_time in raw_scenes:
        s_sec = start_time.seconds
        e_sec = min(max_duration, end_time.seconds)
        if s_sec < e_sec:
            scenes.append((round(s_sec, 2), round(e_sec, 2)))

    return scenes
