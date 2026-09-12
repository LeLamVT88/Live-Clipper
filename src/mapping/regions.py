from __future__ import annotations

import cv2
import numpy as np

from mapping.config import MappingConfig
from mapping.schema import Rect


def crop_region(frame: np.ndarray, region: Rect) -> np.ndarray:
    x0, y0, x1, y1 = region.pixels(frame.shape)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError(f"Configured crop produced an empty image: {region}")
    return crop.copy()


def _mask_intersection(crop: np.ndarray, crop_region_rect: Rect, exclusion: Rect) -> None:
    left = max(crop_region_rect.x, exclusion.x)
    top = max(crop_region_rect.y, exclusion.y)
    right = min(crop_region_rect.x + crop_region_rect.width, exclusion.x + exclusion.width)
    bottom = min(crop_region_rect.y + crop_region_rect.height, exclusion.y + exclusion.height)
    if left >= right or top >= bottom:
        return
    local = Rect(
        (left - crop_region_rect.x) / crop_region_rect.width,
        (top - crop_region_rect.y) / crop_region_rect.height,
        (right - left) / crop_region_rect.width,
        (bottom - top) / crop_region_rect.height,
    )
    x0, y0, x1, y1 = local.pixels(crop.shape)
    cv2.rectangle(crop, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)


def crop_lineup_region(frame: np.ndarray, config: MappingConfig) -> np.ndarray:
    """Crop the starters area and black out configured broadcast overlays."""
    result = crop_region(frame, config.lineup_region)
    exclusions = [*config.exclude_regions]
    if config.substitutes_region is not None:
        exclusions.append(config.substitutes_region)
    if config.coach_region is not None:
        exclusions.append(config.coach_region)
    for exclusion in exclusions:
        _mask_intersection(result, config.lineup_region, exclusion)
    return result

