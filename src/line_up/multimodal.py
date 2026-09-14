from __future__ import annotations

import cv2
import numpy as np


def compute_whole_frame_stationarity(
    frame_t: np.ndarray,
    frame_t2: np.ndarray,
    target_height: int = 180,
) -> float:
    """Compute Whole-Frame Stationarity between frame t and frame t+0.5s without OCR.

    Combines Mean Absolute Difference (MAD) and Normalized Cross-Correlation (NCC).
    Whole-frame motion can include animated portraits or stadium backgrounds;
    a low result alone is not evidence that a lineup is absent.
    """
    h, w = frame_t.shape[:2]
    new_w = int(w * (target_height / h))
    dim = (new_w, target_height)

    gray1 = cv2.cvtColor(cv2.resize(frame_t, dim, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(cv2.resize(frame_t2, dim, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)

    mad = float(np.mean(cv2.absdiff(gray1, gray2)))
    g1_f = gray1.astype(np.float32).flatten()
    g2_f = gray2.astype(np.float32).flatten()

    std1, std2 = np.std(g1_f), np.std(g2_f)
    if std1 < 1e-4 or std2 < 1e-4:
        return 0.0

    corr = float(np.corrcoef(g1_f, g2_f)[0, 1])

    # MAD score: 1.0 if MAD <= 1.0, 0.0 if MAD >= 6.0
    mad_score = np.clip(1.0 - (mad - 1.0) / 5.0, 0.0, 1.0)
    # Correlation score: 1.0 if corr >= 0.98, 0.0 if corr <= 0.70
    corr_score = np.clip((corr - 0.70) / 0.28, 0.0, 1.0)

    stationarity = 0.5 * mad_score + 0.5 * corr_score
    return float(round(stationarity, 4))


def compute_box_scaled_layout_score(boxes: list, frame_shape: tuple[int, ...]) -> float:
    """Evaluate spatial layout signature of football lineups scaled by box count.

    Lineup cards contain vertically aligned columns of player names & numbers (8-11 items).
    Scales with total box count to reject coincidental 2-3 box alignments.
    """
    box_count = len(boxes)
    if box_count < 6:
        return 0.0

    h, w = frame_shape[:2]

    centers = []
    for b in boxes:
        pts = np.array(b, dtype=np.float32)
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        bh = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
        centers.append((cx, cy, bh))

    centers.sort(key=lambda item: item[1])

    # Cluster boxes into vertical columns (similar cx within 12% of screen width)
    x_tol = w * 0.12
    columns: list[list[tuple[float, float, float]]] = []

    for item in centers:
        cx, cy, bh = item
        matched_col = False
        for col in columns:
            col_cx_avg = np.mean([p[0] for p in col])
            if abs(cx - col_cx_avg) <= x_tol:
                col.append(item)
                matched_col = True
                break
        if not matched_col:
            columns.append([item])

    max_aligned = max(len(col) for col in columns) if columns else 0

    # Alignment reaches full weight at nine boxes in a column.
    align_quality = np.clip((max_aligned - 3.0) / 6.0, 0.0, 1.0)
    # Density reaches full weight at seventeen total boxes.
    density_scale = np.clip((box_count - 5.0) / 12.0, 0.0, 1.0)

    score = align_quality * density_scale
    return float(round(score, 4))


def compute_combined_multimodal_score(
    s_ocr: float,
    s_frame_static: float | None,
    s_layout_scaled: float,
    *,
    strong_evidence: bool,
) -> float:
    """Use CV to reject weak moving overlays; never veto strong OCR evidence.

    Static tables can resemble lineups, so CV cannot promote a semantic negative.
    Missing paired frames leave the semantic score unchanged.
    """
    if strong_evidence or s_frame_static is None:
        return s_ocr
    if s_frame_static < 0.45 and s_layout_scaled < 0.50:
        return round(min(s_ocr, 0.10), 4)
    return s_ocr
