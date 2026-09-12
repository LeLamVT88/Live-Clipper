from __future__ import annotations

import cv2
import numpy as np

from mapping.schema import FrameSample, StableSegment


def frame_difference(left: np.ndarray, right: np.ndarray, target_width: int = 320) -> float:
    """Return mean absolute pixel difference on small grayscale frames."""
    if left.size == 0 or right.size == 0:
        raise ValueError("Cannot compare empty frames")

    def prepare(image: np.ndarray) -> np.ndarray:
        height = max(1, round(image.shape[0] * target_width / image.shape[1]))
        return cv2.cvtColor(
            cv2.resize(image, (target_width, height), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )

    a, b = prepare(left), prepare(right)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    return float(np.mean(cv2.absdiff(a, b)))


def find_stable_segments(
    frames: list[FrameSample], max_mean_diff: float = 2.8, min_frames: int = 3,
) -> list[StableSegment]:
    if min_frames < 2:
        raise ValueError("min_frames must be >= 2")
    if len(frames) < min_frames:
        return []
    diffs = [frame_difference(frames[index - 1].image, frames[index].image)
             for index in range(1, len(frames))]
    segments: list[StableSegment] = []
    start: int | None = None
    for diff_index, difference in enumerate(diffs):
        if difference <= max_mean_diff:
            if start is None:
                start = diff_index
            continue
        if start is not None:
            end = diff_index
            if end - start + 1 >= min_frames:
                values = diffs[start:end]
                segments.append(StableSegment(
                    start, end, frames[start].timestamp, frames[end].timestamp,
                    float(np.mean(values)),
                ))
            start = None
    if start is not None:
        end = len(frames) - 1
        if end - start + 1 >= min_frames:
            segments.append(StableSegment(
                start, end, frames[start].timestamp, frames[end].timestamp,
                float(np.mean(diffs[start:end])),
            ))
    return segments


def stable_frames(frames: list[FrameSample], segments: list[StableSegment]) -> list[FrameSample]:
    indexes = {index for segment in segments for index in range(segment.start_index, segment.end_index + 1)}
    return [frame for index, frame in enumerate(frames) if index in indexes]


def select_representative_frame(
    frames: list[FrameSample], segments: list[StableSegment], start: float, end: float,
) -> FrameSample | None:
    candidates = [segment for segment in segments
                  if segment.end_seconds >= start and segment.start_seconds <= end]
    if not candidates:
        return None
    longest = max(candidates, key=lambda item: (
        min(end, item.end_seconds) - max(start, item.start_seconds), -item.mean_diff,
    ))
    eligible = [frame for frame in frames[longest.start_index:longest.end_index + 1]
                if start <= frame.timestamp <= end]
    return eligible[len(eligible) // 2] if eligible else None

