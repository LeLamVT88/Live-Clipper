from __future__ import annotations

from statistics import median

import cv2
import numpy as np

from mapping.config import PlayerDetectionConfig
from mapping.schema import PlayerUnit


def _iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    if not intersection:
        return 0.0
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / max(1, left_area + right_area - intersection)


def sort_player_units(units: list[PlayerUnit], tolerance_ratio: float = 0.55) -> list[PlayerUnit]:
    if not units:
        return []
    typical_height = median(unit.shirt_box[3] - unit.shirt_box[1] for unit in units)
    tolerance = max(5.0, typical_height * tolerance_ratio)
    rows: list[list[PlayerUnit]] = []
    for unit in sorted(units, key=lambda item: ((item.shirt_box[1] + item.shirt_box[3]) / 2,
                                                (item.shirt_box[0] + item.shirt_box[2]) / 2)):
        center_y = (unit.shirt_box[1] + unit.shirt_box[3]) / 2
        row = next((items for items in rows if abs(
            median((item.shirt_box[1] + item.shirt_box[3]) / 2 for item in items) - center_y
        ) <= tolerance), None)
        if row is None:
            rows.append([unit])
        else:
            row.append(unit)
    return [unit for row in rows for unit in sorted(row, key=lambda item: item.shirt_box[0])]


def detect_player_blobs(
    lineup_image: np.ndarray, config: PlayerDetectionConfig | None = None,
) -> list[PlayerUnit]:
    """Detect contrasting shirt blobs and expand each into a number/name player unit."""
    cfg = config or PlayerDetectionConfig()
    if lineup_image.size == 0:
        return []
    height, width = lineup_image.shape[:2]
    hsv = cv2.cvtColor(lineup_image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(lineup_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
    background = np.median(lab.reshape(-1, 3), axis=0)
    colour_distance = np.linalg.norm(lab - background, axis=2)
    contrast = colour_distance >= cfg.background_color_distance
    coloured = (saturation >= cfg.min_saturation) & (value >= cfg.min_value)
    white = (saturation <= cfg.white_saturation_max) & (value >= cfg.white_value_min)
    dark = value <= max(35, cfg.min_value)
    mask = np.where(contrast & (coloured | white | dark), 255, 0).astype(np.uint8)
    kernel_size = max(1, int(cfg.morphology_kernel) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = height * width
    candidates: list[tuple[tuple[int, int, int, int], float]] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        edge_x = round(width * cfg.edge_margin_pct)
        edge_y = round(height * cfg.edge_margin_pct)
        if (x <= edge_x or y <= edge_y or x + box_width >= width - edge_x
                or y + box_height >= height - edge_y):
            continue
        area_ratio = cv2.contourArea(contour) / max(1, frame_area)
        aspect = box_width / max(1, box_height)
        if not cfg.min_area_pct <= area_ratio <= cfg.max_area_pct:
            continue
        if not cfg.min_aspect_ratio <= aspect <= cfg.max_aspect_ratio:
            continue
        fill = cv2.contourArea(contour) / max(1, box_width * box_height)
        if fill < 0.28:
            continue
        candidates.append(((x, y, x + box_width, y + box_height), float(area_ratio * fill)))

    kept: list[tuple[tuple[int, int, int, int], float]] = []
    for box, score in sorted(candidates, key=lambda item: item[1], reverse=True):
        if not any(_iou(box, existing) >= 0.45 for existing, _ in kept):
            kept.append((box, score))

    units = []
    for shirt, score in kept:
        x0, y0, x1, y1 = shirt
        shirt_width, shirt_height = x1 - x0, y1 - y0
        horizontal = round(shirt_width * cfg.horizontal_extension_ratio)
        name_height = round(shirt_height * cfg.name_extension_ratio)
        unit_box = (max(0, x0 - horizontal), y0, min(width, x1 + horizontal), min(height, y1 + name_height))
        number_box = (
            x0 + round(shirt_width * .20), y0 + round(shirt_height * .12),
            x1 - round(shirt_width * .20), y0 + round(shirt_height * .78),
        )
        name_box = (max(0, x0 - horizontal), y1, min(width, x1 + horizontal), min(height, y1 + name_height))
        units.append(PlayerUnit(unit_box, shirt, number_box, name_box, score))
    return sort_player_units(units, cfg.row_tolerance_ratio)


def crop_box(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return image[max(0, y0):min(image.shape[0], y1), max(0, x0):min(image.shape[1], x1)].copy()
