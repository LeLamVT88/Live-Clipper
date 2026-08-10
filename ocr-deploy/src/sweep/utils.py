import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging

def setup_logger(output_dir):
    """Sets up a logger to write to both stdout and a log file."""
    # Reset existing handlers to prevent duplicate logging
    logger = logging.getLogger('goal_detector')
    
    # Configure logging level based on DEBUG env variable
    debug_mode = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes", "y")
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logger.setLevel(log_level)
    logger.handlers.clear()
    
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # File handler (only in debug mode)
    if debug_mode:
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, 'run.log')
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        fh.setLevel(log_level)
        logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(log_level)
    logger.addHandler(ch)
    
    return logger

def preprocess_for_ocr(crop_img, upscale_factor=3):
    """
    Applies the preprocessing pipeline to the score crop for OCR:
    1. Convert to grayscale
    2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    3. Upscale (3x default) using cubic interpolation
    4. Denoise
    5. Generate normal and inverted binary versions using Otsu thresholding
    6. Generate normal and inverted binary versions using Adaptive thresholding
    """
    if len(crop_img.shape) == 3:
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop_img.copy()
        
    # 2. CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    
    # 3. Upscale
    h, w = gray.shape[:2]
    gray_up = cv2.resize(gray, (w * upscale_factor, h * upscale_factor), interpolation=cv2.INTER_CUBIC)
    
    # 4. Denoise
    # Gaussian blur is fast and effective for Tesseract pre-filtering
    denoised = cv2.GaussianBlur(gray_up, (3, 3), 0)
    
    # 5. Otsu thresholding
    _, otsu_norm = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_inv = cv2.bitwise_not(otsu_norm)
    
    return [
        ("otsu_norm", otsu_norm),
        ("otsu_inv", otsu_inv)
    ]

def save_variance_heatmap(heatmap, filepath, title="Temporal Variance Heatmap"):
    """Saves the coarse block variance heatmap using matplotlib."""
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Variance')
    plt.title(title)
    plt.xlabel('Block X')
    plt.ylabel('Block Y')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()

def save_combined_score_map(score_map, filepath, title="Combined Score Map (Gradient / Variance)"):
    """Saves the combined score map (Stage 1 localisation score)."""
    plt.figure(figsize=(8, 6))
    plt.imshow(score_map, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Score')
    plt.title(title)
    plt.xlabel('Block X')
    plt.ylabel('Block Y')
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()

def draw_and_save_bbox(frame, bbox, filepath, title="Localised Region", color=(0, 255, 0), thickness=2):
    """Draws a bounding box on the frame and saves it."""
    out_img = frame.copy()
    x, y, w, h = bbox
    cv2.rectangle(out_img, (x, y), (x + w, y + h), color, thickness)
    # Put label
    cv2.putText(out_img, title, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(filepath, out_img)

def save_isolated_subcrops(scoreboard_crop, score_bbox, timer_bbox, filepath):
    """
    Saves a visualization of the isolated regions within the scoreboard crop.
    score_bbox: (x, y, w, h) relative to the scoreboard crop
    timer_bbox: (x, y, w, h) relative to the scoreboard crop or None
    """
    vis = scoreboard_crop.copy()
    
    # Draw score bbox (Green)
    sx, sy, sw, sh = score_bbox
    cv2.rectangle(vis, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
    cv2.putText(vis, "Score", (sx, max(sy - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Draw timer bbox (Red) if present
    if timer_bbox is not None:
        tx, ty, tw, th = timer_bbox
        cv2.rectangle(vis, (tx, ty), (tx + tw, ty + th), (0, 0, 255), 2)
        cv2.putText(vis, "Timer", (tx, max(ty - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
    cv2.imwrite(filepath, vis)

def save_goal_collage(before_frame, after_frame, before_score, after_score, timestamp_sec, filepath):
    """
    Saves a collage of the goal event:
    Shows the frame before and after the goal side-by-side, with overlay details.
    """
    h, w = before_frame.shape[:2]
    # Resize frames to half size for compact saving
    w_new, h_new = w // 2, h // 2
    b_resized = cv2.resize(before_frame, (w_new, h_new))
    a_resized = cv2.resize(after_frame, (w_new, h_new))
    
    # Create side-by-side collage
    collage = np.hstack((b_resized, a_resized))
    
    # Add black border at the bottom for text info
    border_h = 60
    collage_with_text = cv2.copyMakeBorder(
        collage, 0, border_h, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    
    # Render information text
    min_val = int(timestamp_sec // 60)
    sec_val = int(timestamp_sec % 60)
    info_text = f"Goal Event at {min_val:02d}:{sec_val:02d} ({timestamp_sec:.1f}s) | Score: {before_score} -> {after_score}"
    
    cv2.putText(
        collage_with_text,
        info_text,
        (20, h_new + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    
    cv2.imwrite(filepath, collage_with_text)
