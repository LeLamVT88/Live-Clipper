import av
import cv2
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import tempfile

from collections import defaultdict, deque
from paddleocr import PaddleOCR, TextRecognition
from tqdm import tqdm
import subprocess

from src.config import GOAL_DETECTOR_DEFAULT_CONFIG
from src.replay.detector import ReplayDetector
from src.sweep.utils import (
    save_variance_heatmap,
    save_combined_score_map,
    draw_and_save_bbox,
    save_isolated_subcrops,
    save_goal_collage,
)

# Bypass PaddleX connectivity check for models
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
logging.getLogger("ppocr").setLevel(logging.ERROR)

# Redirect Tesseract's temp files to RAM (tmpfs) to avoid /tmp/ disk I/O spam.
os.environ.setdefault("TMPDIR", "/dev/shm")

logger = logging.getLogger("goal_detector")


class CalibrationError(Exception):
    """Custom exception raised when post-calibration validation fails."""

    pass


class GoalDetector:
    _shared_ocr_pipeline = None
    _shared_ocr_rec = None

    def __init__(self, video_path, output_dir, config=None):
        self.video_path = video_path
        self.output_dir = output_dir

        # Load configuration with defaults
        self.config = GOAL_DETECTOR_DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        # ======DEBUG======
        # Setup debug directory if requested (either via config or via DEBUG env var)
        debug_env = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes", "y")
        if self.config.get("debug_calibration", False) or debug_env:
            base_dir = (
                "/dev/shm"
                if os.path.exists("/dev/shm") and os.access("/dev/shm", os.W_OK)
                else tempfile.gettempdir()
            )
            self.debug_dir = os.path.join(base_dir, "ocr_debug")
        else:
            self.debug_dir = None
        if self.debug_dir:
            import shutil

            if os.path.exists(self.debug_dir):
                shutil.rmtree(self.debug_dir)
            os.makedirs(self.debug_dir, exist_ok=True)
            os.makedirs(os.path.join(self.debug_dir, "stage0_cleaning"), exist_ok=True)
            os.makedirs(os.path.join(self.debug_dir, "stage0_presence"), exist_ok=True)
            os.makedirs(
                os.path.join(self.debug_dir, "stage1_localization"), exist_ok=True
            )
            os.makedirs(os.path.join(self.debug_dir, "stage2_isolation"), exist_ok=True)
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(
                f"Calibration debug mode active. Exporting visualizations to {self.debug_dir}"
            )
        # ======DEBUG======

        # State variables
        self.state = "CALIBRATING"
        self.scoreboard_bbox = None  # (x, y, w, h) absolute
        self.is_split_score = False  # Track whether score is split into two boxes

        # Unified score mode coordinates
        self.score_bbox_rel = None  # (x, y, w, h) relative to scoreboard_bbox
        self.score_bbox_abs = None  # (x, y, w, h) absolute (padded)

        # Split score mode coordinates
        self.home_score_bbox_rel = None
        self.away_score_bbox_rel = None
        self.home_score_bbox_abs = None
        self.away_score_bbox_abs = None

        self.timer_bbox_rel = None  # (x, y, w, h) relative to scoreboard_bbox
        self.timer_bbox_abs = None  # (x, y, w, h) absolute

        # Non-OCR scoreboard visibility templates
        self.scoreboard_template = None
        self.scoreboard_mask = None
        self.scoreboard_threshold = None

        self.debounce_buffer = deque(maxlen=self.config["debounce_k"])
        self.last_accepted_score = None
        self.absent_counter = 0

        # Caching variables for fast tracking
        self.last_home_crop = None
        self.last_away_crop = None
        self.last_score_crop = None
        self.last_reading = None
        self.ocr_run_count = 0
        self.ocr_skip_count = 0

        # Initialize PaddleOCR pipelines (PP-OCRv6 family, medium tier for detection)
        if GoalDetector._shared_ocr_pipeline is None:
            logger.info("Initializing PaddleOCR pipelines...")
            GoalDetector._shared_ocr_pipeline = PaddleOCR(
                lang="en",
                text_detection_model_name="PP-OCRv6_medium_det",
                text_recognition_model_name="PP-OCRv6_medium_rec",
                enable_mkldnn=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            GoalDetector._shared_ocr_rec = TextRecognition(model_name="PP-OCRv6_tiny_rec")
            logger.info("PaddleOCR pipelines initialized successfully.")
        
        self._ocr_pipeline = GoalDetector._shared_ocr_pipeline
        self._ocr_rec = GoalDetector._shared_ocr_rec

        # Video properties
        self.fps = 0.0
        self.total_frames = 0
        self.width = 0
        self.height = 0
        self.duration = 0.0

        self._inspect_video()

    def _inspect_video(self):
        with av.open(self.video_path) as container:
            video_stream = container.streams.video[0]
            self.fps = float(video_stream.average_rate)

            # Get duration
            if video_stream.duration:
                self.duration = float(video_stream.duration * video_stream.time_base)
            else:
                self.duration = (
                    float(container.duration / 1000000.0) if container.duration else 0.0
                )

            # Get frame count
            if video_stream.frames:
                self.total_frames = video_stream.frames
            else:
                self.total_frames = (
                    int(round(self.fps * self.duration)) if self.fps > 0 else 0
                )

            self.width = video_stream.width
            self.height = video_stream.height
        logger.info(
            f"Video loaded (PyAV): {self.width}x{self.height} @ {self.fps:.2f} fps, duration: {self.duration:.2f}s, total frames: {self.total_frames}"
        )

    def _infer_score_layout_from_tokens(self, score_tokens, bw, bh):
        """
        Infer whether scoreboard score is split or unified using score tokens.

        Returns dict:
        - split mode:
            {
            'mode': 'split',
            'home_bbox_rel': (...),
            'away_bbox_rel': (...)
            }
        - unified mode:
            {
            'mode': 'unified',
            'score_bbox_rel': (...)
            }
        """
        if not score_tokens:
            return None

        # Prefer split if we can find 2 horizontally separated tokens on same row
        best_pair = None
        best_pair_score = -1e18

        mid = bw / 2.0
        # Dynamically adjust center anchor if an asymmetric timer is localized
        if self.timer_bbox_rel is not None:
            tx, ty, tw, th = self.timer_bbox_rel
            # Only shift if timer is clearly on one side of the scoreboard
            if tx + tw <= bw * 0.45:
                # Timer is on the left; shift mid to the center of the remaining space
                non_timer_start = tx + tw
                mid = non_timer_start + (bw - non_timer_start) / 2.0
            elif tx >= bw * 0.55:
                # Timer is on the right; shift mid to the center of the preceding space
                mid = tx / 2.0

        for i in range(len(score_tokens)):
            for j in range(i + 1, len(score_tokens)):
                t1 = score_tokens[i]
                t2 = score_tokens[j]

                b1 = t1["bbox"]
                b2 = t2["bbox"]

                c1x = b1[0] + b1[2] / 2.0
                c2x = b2[0] + b2[2] / 2.0
                c1y = b1[1] + b1[3] / 2.0
                c2y = b2[1] + b2[3] / 2.0

                # same-ish row
                y_diff = abs(c1y - c2y)
                if y_diff > self.config["score_row_tol"] * max(b1[3], b2[3]):
                    continue

                # need some horizontal separation
                if abs(c2x - c1x) < max(10, self.config["score_split_sep_frac"] * bw):
                    continue

                # reward symmetry around center, reward confidence
                pair_center = (c1x + c2x) / 2.0
                center_penalty = abs(pair_center - mid)
                conf_score = float(t1["conf"]) + float(t2["conf"])

                # if the pair straddles the center, that's even better
                straddle_bonus = 0.0
                if c1x < mid < c2x:
                    straddle_bonus = self.config["score_straddle_bonus"]

                total_score = (
                    conf_score
                    + straddle_bonus
                    - self.config["score_center_penalty_w"] * center_penalty
                    - self.config["score_y_penalty_w"] * y_diff
                )

                if total_score > best_pair_score:
                    best_pair_score = total_score
                    best_pair = (t1, t2)

        if best_pair is not None:
            left_tok, right_tok = best_pair
            b1 = tuple(int(c) for c in left_tok["bbox"])
            b2 = tuple(int(c) for c in right_tok["bbox"])
            return {"mode": "split", "home_bbox_rel": b1, "away_bbox_rel": b2}

        # Fallback: unified score = token closest to scoreboard center
        best_tok = min(
            score_tokens,
            key=lambda t: abs((t["bbox"][0] + t["bbox"][2] / 2.0) - mid),
        )
        return {
            "mode": "unified",
            "score_bbox_rel": tuple(int(c) for c in best_tok["bbox"]),
        }

    def _find_score_tokens_excluding_timer(self, all_tokens, timer_bbox_rel, bw, bh):
        """
        Collect digit-like tokens that are NOT part of timer.
        These become score candidates.
        """
        score_tokens = []

        max_digit_w = max(60, int(self.config["digit_max_w_frac"] * bw))
        max_digit_h = max(20, int(self.config["digit_max_h_frac"] * bh))

        for tok in all_tokens:
            digit = self._normalize_ocr_token(tok["text"])
            if not (digit.isdigit() and 1 <= len(digit) <= 2):
                continue

            tx, ty, tw, th = tok["bbox"]

            # size sanity
            if tw <= 0 or th <= 0:
                continue
            if tw > max_digit_w or th > max_digit_h:
                continue

            # if timer exists, reject tokens that overlap it
            if timer_bbox_rel is not None:
                ov = self._overlap_ratio(tok["bbox"], timer_bbox_rel)
                if ov > self.config["timer_overlap_reject"]:
                    continue

            score_tokens.append(tok)

        # left-to-right order
        score_tokens.sort(key=lambda t: t["bbox"][0])
        return score_tokens

    def _find_timer_from_tokens(self, all_tokens, bw, bh):
        """
        Find timer token group from OCR tokens using colon-first logic.

        Priority:
        1) Direct token like '12:34'
        2) Assemble from [digits] + ':' + [digits]
        3) Fallback compact timer token like '1234' -> 12:34 or '712' -> 7:12
        """
        # ---------- Pass 1: direct full timer token ----------
        direct_candidates = []
        for tok in all_tokens:
            t = self._normalize_ocr_token(tok["text"])
            if re.fullmatch(r"\d{1,3}:\d{2}", t):
                parts = t.split(":")
                mins = int(parts[0])
                secs = int(parts[1])
                if 0 <= mins <= self.config["timer_max_minutes"] and 0 <= secs <= self.config["timer_max_seconds"]:
                    direct_candidates.append(tok)

        if direct_candidates:
            best = max(direct_candidates, key=lambda z: z["conf"])
            bbox = tuple(int(c) for c in best["bbox"])
            return {
                "kind": "direct",
                "tokens": [best],
                "bbox": bbox,
                "text": self._normalize_ocr_token(best["text"]),
            }

        # ---------- Pass 2: assemble from colon anchor ----------
        colon_candidates = []
        for tok in all_tokens:
            t = self._normalize_ocr_token(tok["text"])

            # True colon or OCR-confused colon-like token
            if ":" in t or t in [";", ".", "::"]:
                colon_candidates.append(tok)

        best_candidate = None
        best_score = -1e18

        x_gap_max = max(self.config["timer_colon_gap_min_px"], int(self.config["timer_colon_gap_frac"] * bw))  # allowed horizontal gap around ':'

        for colon_tok in colon_candidates:
            cb = colon_tok["bbox"]
            cx1 = cb[0]
            cx2 = cb[0] + cb[2]

            left_candidates = []
            right_candidates = []

            for tok in all_tokens:
                if tok is colon_tok:
                    continue
                if not self._is_digit_token(tok["text"]):
                    continue

                tb = tok["bbox"]
                tx1 = tb[0]
                tx2 = tb[0] + tb[2]

                # Must roughly lie on same row as ':'
                if self._vertical_overlap_ratio(tb, cb) < self.config["timer_voverlap_min"]:
                    continue

                # left of colon
                if tx2 <= cx1:
                    gap = cx1 - tx2
                    if gap <= x_gap_max:
                        left_candidates.append((gap, tok))

                # right of colon
                if tx1 >= cx2:
                    gap = tx1 - cx2
                    if gap <= x_gap_max:
                        right_candidates.append((gap, tok))

            left_candidates.sort(key=lambda z: z[0])
            right_candidates.sort(key=lambda z: z[0])

            # Try best few candidates on each side
            for _, lt in left_candidates[: self.config["timer_candidate_pairs"]]:
                for _, rt in right_candidates[: self.config["timer_candidate_pairs"]]:
                    lt_text = self._normalize_ocr_token(lt["text"])
                    rt_text = self._normalize_ocr_token(rt["text"])

                    if not (lt_text.isdigit() and rt_text.isdigit()):
                        continue

                    mins = int(lt_text)
                    secs = int(rt_text)

                    # football time validity
                    if not (0 <= mins <= self.config["timer_max_minutes"] and 0 <= secs <= self.config["timer_max_seconds"]):
                        continue

                    # score the candidate:
                    # high OCR confidence + small gap to colon + decent vertical alignment
                    lt_box = lt["bbox"]
                    rt_box = rt["bbox"]

                    left_gap = abs(cx1 - (lt_box[0] + lt_box[2]))
                    right_gap = abs(rt_box[0] - cx2)

                    row_score = self._vertical_overlap_ratio(
                        lt_box, cb
                    ) + self._vertical_overlap_ratio(rt_box, cb)

                    conf_score = (
                        float(lt["conf"]) + float(rt["conf"]) + float(colon_tok["conf"])
                    )
                    geometry_penalty = self.config["timer_gap_penalty_w"] * (left_gap + right_gap)

                    total_score = conf_score + self.config["timer_row_score_w"] * row_score - geometry_penalty

                    if total_score > best_score:
                        best_score = total_score
                        timer_text = f"{mins:02d}:{secs:02d}"
                        timer_bbox = self._union_boxes([lt_box, cb, rt_box])
                        best_candidate = {
                            "kind": "assembled",
                            "tokens": [lt, colon_tok, rt],
                            "bbox": timer_bbox,
                            "text": timer_text,
                        }

        if best_candidate is not None:
            return best_candidate

        # ---------- Pass 3: compact token fallback ----------
        # e.g. "1234" -> 12:34 or "712" -> 7:12
        compact_candidates = []
        for tok in all_tokens:
            t = self._normalize_ocr_token(tok["text"])
            if re.fullmatch(r"\d{3,4}", t):
                mins = int(t[:-2])
                secs = int(t[-2:])
                if 0 <= mins <= self.config["timer_max_minutes"] and 0 <= secs <= self.config["timer_max_seconds"]:
                    compact_candidates.append(tok)

        if compact_candidates:
            best = max(compact_candidates, key=lambda z: z["conf"])
            t = self._normalize_ocr_token(best["text"])
            mins = int(t[:-2])
            secs = int(t[-2:])
            bbox = tuple(int(c) for c in best["bbox"])
            return {
                "kind": "compact",
                "tokens": [best],
                "bbox": bbox,
                "text": f"{mins:02d}:{secs:02d}",
            }

        return None

    def _normalize_ocr_token(self, text):
        """Normalize OCR text for timer/score parsing."""
        if text is None:
            return ""
        t = str(text).strip()
        t = t.replace("O", "0").replace("o", "0")
        t = t.replace("I", "1").replace("l", "1")
        t = t.replace("S", "5").replace("s", "5")
        t = re.sub(r"\s+", "", t)
        return t

    def _is_digit_token(self, text):
        """Returns True if token is 1-2 digits after normalization."""
        t = self._normalize_ocr_token(text)
        return t.isdigit() and 1 <= len(t) <= 2

    def _vertical_overlap_ratio(self, box1, box2):
        """Vertical overlap ratio normalized by smaller height."""
        y1a, y1b = box1[1], box1[1] + box1[3]
        y2a, y2b = box2[1], box2[1] + box2[3]
        inter = max(0, min(y1b, y2b) - max(y1a, y2a))
        denom = max(1e-6, min(box1[3], box2[3]))
        return inter / denom

    def _union_boxes(self, boxes):
        """Union of a list of (x,y,w,h) boxes."""
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[0] + b[2] for b in boxes)
        y2 = max(b[1] + b[3] for b in boxes)
        return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

    def _overlap_ratio(self, box1, box2):
        """
        Overlap ratio normalized by smaller area.
        Useful for asking: does token box belong to timer box?
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = max(1, box1[2] * box1[3])
        area2 = max(1, box2[2] * box2[3])
        return inter / max(1, min(area1, area2))

    def _pad_rel_box(self, box, bw, bh, pad=2):
        """Pad a relative box but keep it inside scoreboard crop bounds."""
        x, y, w, h = box
        px = max(0, int(x - pad))
        py = max(0, int(y - pad))
        pw = min(bw - px, int(w + 2 * pad))
        ph = min(bh - py, int(h + 2 * pad))
        return (px, py, pw, ph)

    def _read_score_from_frame(self, img, min_conf=None):
        """Runs OCR on the given frame to parse the score."""
        if not self._is_scoreboard_present(img):
            return None
        if self.is_split_score:
            if self.home_score_bbox_abs is None or self.away_score_bbox_abs is None:
                return None
            hx, hy, hw, hh = self.home_score_bbox_abs
            ax, ay, aw, ah = self.away_score_bbox_abs
            return self._perform_ocr(
                img[hy : hy + hh, hx : hx + hw],
                img[ay : ay + ah, ax : ax + aw],
                min_conf=min_conf,
            )
        else:
            if self.score_bbox_abs is None:
                return None
            sx, sy, sw, sh = self.score_bbox_abs
            return self._perform_ocr(img[sy : sy + sh, sx : sx + sw], min_conf=min_conf)

    def _validate_frame_relaxed(self, frame):
        """
        Sanity checks the scoreboard region using robust full-crop OCR:
        1) Detects text in scoreboard crop using robust _ocr_pipeline.
        2) Strips out the timer tokens (containing colon).
        3) Verifies there are at least 2 remaining digits in other tokens.
        """
        if self.scoreboard_bbox is None:
            return False
        bx, by, bw, bh = self.scoreboard_bbox
        fh, fw = frame.shape[:2]
        bx = max(0, min(bx, fw - 1))
        by = max(0, min(by, fh - 1))
        bw = max(1, min(bw, fw - bx))
        bh = max(1, min(bh, fh - by))

        scoreboard_crop = frame[by : by + bh, bx : bx + bw]
        try:
            res_list = list(self._ocr_pipeline.predict(scoreboard_crop))
            if not res_list or len(res_list) == 0:
                return False

            res = res_list[0]
            texts = res.get("rec_texts", []) or []

            # Filter out any timer-like tokens (containing colon, semicolons, etc.)
            non_timer_digits = []
            for text in texts:
                text_norm = self._normalize_ocr_token(text)
                if any(delim in text_norm for delim in (":", ";", ".")):
                    continue
                # Remove non-digits to isolate only digits
                digits = re.sub(r"\D", "", text_norm)
                if digits:
                    non_timer_digits.append(digits)

            joined_digits = "".join(non_timer_digits)
            # Need at least 2 digits overall for the scores (e.g. 0 and 0)
            if len(joined_digits) >= 2:
                return True
        except Exception as e:
            logger.warning(f"Error in relaxed validation: {e}")
        return False

    def calibrate(self):
        """Runs Stage 0 to Stage 3 to locate the scoreboard and score box, and performs validation."""
        logger.info("Starting calibration (Stages 0–3)...")

        # --- Stage 0: Frame Sampling and Selection ---
        sampled_frames = self._sample_frames()
        clean_frames = self._filter_clean_frames(sampled_frames)
        clean_frames = self._filter_scoreboard_present_frames(clean_frames)

        # --- Stage 1: Scoreboard Localisation ---
        self.scoreboard_bbox = self._localise_scoreboard(clean_frames)

        # Partition clean_frames for validation holdout from the middle 80% of the video
        val_count = self.config["validation_holdout_count"]
        held_out_frames = []
        val_indices = set()

        start_idx = len(clean_frames) // 10
        end_idx = 9 * len(clean_frames) // 10

        if len(clean_frames) >= val_count * 2 and val_count > 0:
            interval_len = (end_idx - start_idx) // val_count
            for i in range(val_count):
                sub_start = start_idx + i * interval_len
                sub_end = sub_start + interval_len
                found = False
                for idx in range(sub_start, sub_end):
                    frame = clean_frames[idx]
                    if self._validate_frame_relaxed(frame):
                        logger.info(f"Verified frame index {idx} as eligible validation frame.")
                        held_out_frames.append(frame)
                        val_indices.add(idx)
                        found = True
                        break
                    else:
                        logger.warning(f"Frame index {idx} failed relaxed validation check (no timer or missing score digits).")
                if not found:
                    logger.warning(f"Could not find verified frame in sub-interval #{i+1} [{sub_start}, {sub_end}]. Falling back to frame index {sub_start}.")
                    for idx in range(sub_start, sub_end):
                        held_out_frames.append(clean_frames[idx])
                        val_indices.add(idx)
                        break

            import random
            random.shuffle(held_out_frames)

            calibration_frames = [
                clean_frames[i]
                for i in range(len(clean_frames))
                if i not in val_indices
            ]
            logger.info(
                f"Held out {len(held_out_frames)} verified frames for post-calibration validation. Calibration frames: {len(calibration_frames)}"
            )
        else:
            held_out_frames = clean_frames
            calibration_frames = clean_frames
            logger.warning(
                f"Too few clean frames ({len(clean_frames)}) to hold out cleanly. Using all for both calibration & validation."
            )

        # --- Stage 2: Score Region Isolation ---
        self._isolate_score_region(calibration_frames)

        # --- Stage 3: Create Scoreboard Template ---
        self._create_scoreboard_template(calibration_frames)

        # --- Stage 4: Post-Calibration Validation ---
        logger.info("Running post-calibration validation on held-out frames...")
        valid_reads = 0
        attempts = 0
        success_needed = min(self.config["validation_holdout_count"], len(held_out_frames))
        min_success = min(3, success_needed)

        if success_needed <= 0:
            raise CalibrationError("No frames available for validation pool.")

        for idx, frame in enumerate(held_out_frames):
            attempts += 1
            result = self._read_score_from_frame(frame, min_conf=30.0)
            if result is not None:
                h, a = result
                if (
                    0 <= h <= self.config["max_score_value"]
                    and 0 <= a <= self.config["max_score_value"]
                ):
                    valid_reads += 1
                    logger.info(
                        f"Validation frame #{idx+1}: Success, read score {h}-{a} (total successes: {valid_reads}/{min_success})"
                    )
                    if valid_reads >= min_success:
                        logger.info(f"Post-calibration validation passed early (found {min_success} valid readings).")
                        break
                else:
                    logger.warning(
                        f"Validation frame #{idx+1}: Failed (score out of range: {h}-{a})"
                    )
            else:
                logger.warning(
                    f"Validation frame #{idx+1}: Failed (could not read score)"
                )

        if valid_reads < min_success:
            raise CalibrationError(
                f"Post-calibration validation failed ({valid_reads}/{min_success} frames passed verification after checking {attempts} frames). "
                f"Scoreboard bbox: {self.scoreboard_bbox}, Score bbox: "
                f"{self.score_bbox_abs if not self.is_split_score else (self.home_score_bbox_abs, self.away_score_bbox_abs)}"
            )

        self.state = "TRACKING"
        logger.info("Calibration complete. Scoreboard and score crops locked.")
        logger.info(f"Scoreboard Bounding Box: {self.scoreboard_bbox}")
        if self.is_split_score:
            logger.info(
                f"Detected Split Scoreboard. Home Rel: {self.home_score_bbox_rel}, Away Rel: {self.away_score_bbox_rel}"
            )
            logger.info(
                f"Home Abs Padded: {self.home_score_bbox_abs}, Away Abs Padded: {self.away_score_bbox_abs}"
            )
        else:
            logger.info(
                f"Detected Unified Scoreboard. Rel: {self.score_bbox_rel}, Abs: {self.score_bbox_abs}"
            )

    def _sample_frames(self):
        """Samples N evenly spaced frames by seeking directly in the video (extremely memory & CPU efficient)."""
        n = self.config["sample_n_frames"]
        logger.info(f"Sampling {n} frames from the video via direct seeking...")

        container = av.open(self.video_path)
        video_stream = container.streams.video[0]
        time_base = float(video_stream.time_base)

        # Calculate target timestamps in seconds
        target_secs = np.linspace(0, self.duration - 0.1, n)

        sampled = []
        for idx, sec in enumerate(target_secs):
            pts = int(sec / time_base)
            container.seek(pts, stream=video_stream)

            # Read first frame after seek
            for frame in container.decode(video=0):
                img = frame.to_ndarray(format="bgr24")
                frame_idx = int(round(frame.pts * time_base * self.fps))
                sampled.append((frame_idx, img))
                break

        container.close()
        logger.info(f"Successfully sampled {len(sampled)} frames.")
        return sampled

    def _filter_clean_frames(self, sampled_frames):
        """Filters out replay/filler frames using ROI stability."""
        n = len(sampled_frames)
        if n < 3:
            return [f[1] for f in sampled_frames]

        w_roi = int(self.width * self.config["search_roi_w_frac"])
        h_roi = int(self.height * self.config["search_roi_h_frac"])

        roi_diffs = []
        for i in range(n - 1):
            roi_a = cv2.cvtColor(
                sampled_frames[i][1][0:h_roi, 0:w_roi], cv2.COLOR_BGR2GRAY
            ).astype(np.float32)
            roi_b = cv2.cvtColor(
                sampled_frames[i + 1][1][0:h_roi, 0:w_roi], cv2.COLOR_BGR2GRAY
            ).astype(np.float32)
            roi_diffs.append(float(np.mean(cv2.absdiff(roi_a, roi_b))))

        threshold = np.percentile(roi_diffs, 60)
        clean_frames = [
            sampled_frames[i][1] for i in range(n - 1) if roi_diffs[i] < threshold
        ]

        # Diagnostic export for Stage 0 Clean Frames
        # ======DEBUG======
        if self.debug_dir:
            kept_info = []
            dropped_info = []
            for i in range(n - 1):
                score = roi_diffs[i]
                if score < threshold:
                    kept_info.append((i, sampled_frames[i][1], score))
                else:
                    dropped_info.append((i, sampled_frames[i][1], score))
            # Sort kept frames by stability rank ascending
            kept_info.sort(key=lambda x: x[2])
            for rank, (i, frame, score) in enumerate(kept_info):
                filename = f"kept_rank_{rank:02d}_score_{score:.3f}_idx_{i}.png"
                cv2.imwrite(
                    os.path.join(self.debug_dir, "stage0_cleaning", filename), frame
                )
            for i, frame, score in dropped_info:
                filename = f"dropped_idx_{i}_score_{score:.3f}.png"
                cv2.imwrite(
                    os.path.join(self.debug_dir, "stage0_cleaning", filename), frame
                )
        # ======DEBUG======

        logger.info(
            f"ROI-stability filtering complete. Retained {len(clean_frames)} clean frames (out of {n} sampled)."
        )

        if len(clean_frames) < 10:
            logger.warning(
                "Very few clean frames found! Falling back to using all sampled frames."
            )
            return [f[1] for f in sampled_frames]

        return clean_frames

    def _filter_scoreboard_present_frames(self, frames):
        """Drops frames where the scoreboard HUD is likely absent (pre-match, half-time, etc.)."""
        w_roi = int(self.width * self.config["search_roi_w_frac"])
        # Use just the top 10% of the frame height — scoreboard is always here
        h_hud = max(16, int(self.height * 0.10))

        gradient_scores = []
        for idx, frame in enumerate(frames):
            roi = frame[0:h_hud, 0:w_roi]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sobelx**2 + sobely**2)
            grad_score = float(np.mean(grad_mag))
            gradient_scores.append(grad_score)

            # Export gradient visuals
            # ======DEBUG======
            if self.debug_dir:
                grad_norm = cv2.normalize(
                    grad_mag, None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)
                cv2.imwrite(
                    os.path.join(
                        self.debug_dir,
                        "stage0_presence",
                        f"hud_grad_{idx}_val_{grad_score:.1f}.png",
                    ),
                    grad_norm,
                )
            # ======DEBUG======

        if not gradient_scores:
            return frames

        median_grad = float(np.median(gradient_scores))
        GRAD_FLOOR = 20.0  # px/px — genuinely blank/logo frames are well below this
        threshold = min(median_grad * 0.50, GRAD_FLOOR)
        logger.info(
            f"Scoreboard presence filter: median HUD gradient={median_grad:.2f}, "
            f"drop threshold={threshold:.2f}"
        )

        filtered = [f for f, g in zip(frames, gradient_scores) if g >= threshold]

        # Export dropped frames
        # ======DEBUG======
        if self.debug_dir:
            for idx, (frame, g) in enumerate(zip(frames, gradient_scores)):
                if g < threshold:
                    cv2.imwrite(
                        os.path.join(
                            self.debug_dir,
                            "stage0_presence",
                            f"hud_dropped_{idx}_val_{g:.1f}.png",
                        ),
                        frame,
                    )
        # ======DEBUG======

        n_removed = len(frames) - len(filtered)
        if n_removed > 0:
            logger.info(
                f"Scoreboard presence filter: removed {n_removed} frame(s) without HUD. "
                f"Kept {len(filtered)}/{len(frames)}."
            )

        if len(filtered) < 10:
            logger.warning(
                "Too few frames remain after scoreboard presence filter — falling back to all clean frames."
            )
            return frames

        return filtered

    def _localise_scoreboard(self, clean_frames):
        """Locates the scoreboard ROI using block-level temporal variance and spatial gradients."""
        w_roi = int(self.width * self.config["search_roi_w_frac"])
        h_roi = int(self.height * self.config["search_roi_h_frac"])
        b = self.config["block_size"]

        h_blocks = h_roi // b
        w_blocks = w_roi // b

        # 1. Compute block-level mean intensities for all clean frames
        block_means_list = []
        for frame in clean_frames:
            roi = frame[0:h_roi, 0:w_roi]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Crop to multiple of block size
            gray_roi_crop = gray_roi[: h_blocks * b, : w_blocks * b]
            # Reshape and average
            blocks = gray_roi_crop.reshape(h_blocks, b, w_blocks, b)
            means = blocks.mean(axis=(1, 3))
            block_means_list.append(means)

        block_means_stack = np.stack(
            block_means_list, axis=0
        )  # shape: (|F|, h_blocks, w_blocks)

        # Compute temporal variance of block means
        block_variance = np.var(
            block_means_stack, axis=0
        )  # shape: (h_blocks, w_blocks)
        # ======DEBUG======
        if self.debug_dir:
            save_variance_heatmap(
                block_variance,
                os.path.join(self.output_dir, "calibration_variance_heatmap.png"),
            )
            save_variance_heatmap(
                block_variance,
                os.path.join(
                    self.debug_dir, "stage1_localization", "block_variance.png"
                ),
                title="Stage 1 Block Variance",
            )
        # ======DEBUG======

        # 2. Compute spatial gradients averaged across 5 clean frames
        sample_indices = np.linspace(
            0, len(clean_frames) - 1, min(5, len(clean_frames)), dtype=int
        )
        gradient_maps = []
        for idx in sample_indices:
            frame_roi = clean_frames[idx][0:h_roi, 0:w_roi]
            gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_maps.append(np.sqrt(sobelx**2 + sobely**2))

        mean_gradient = np.mean(gradient_maps, axis=0)
        mean_gradient_crop = mean_gradient[: h_blocks * b, : w_blocks * b]
        sobel_blocks = mean_gradient_crop.reshape(h_blocks, b, w_blocks, b)
        block_gradient = sobel_blocks.mean(axis=(1, 3))

        # ======DEBUG======
        if self.debug_dir:
            # Save raw mean gradient normalized
            mean_grad_norm = cv2.normalize(
                mean_gradient, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
            cv2.imwrite(
                os.path.join(
                    self.debug_dir, "stage1_localization", "mean_gradient.png"
                ),
                mean_grad_norm,
            )
            # Save block average gradient heatmap
            plt.figure(figsize=(8, 6))
            plt.imshow(block_gradient, cmap="hot", interpolation="nearest")
            plt.colorbar(label="Gradient")
            plt.title("Stage 1 Block Spatial Gradient")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    self.debug_dir, "stage1_localization", "block_gradient.png"
                ),
                dpi=150,
            )
            plt.close()
        # ======DEBUG======

        # For drawing bounding box fallback/validation
        mid_frame = clean_frames[len(clean_frames) // 2]

        # 3. Combine signals: score = spatial_gradient / (block_variance + epsilon)
        epsilon = 1e-5
        score_map = block_gradient / (block_variance + epsilon)
        # ======DEBUG======
        if self.debug_dir:
            save_combined_score_map(
                score_map,
                os.path.join(self.output_dir, "calibration_combined_score_map.png"),
            )
            save_combined_score_map(
                score_map,
                os.path.join(self.debug_dir, "stage1_localization", "score_map.png"),
                title="Stage 1 Combined Score Map",
            )
        # ======DEBUG======

        # 4. Bounding Box Extraction
        thresh_val = np.percentile(score_map, 100 - self.config["variance_percentile"])
        binary_map = (score_map >= thresh_val).astype(np.uint8) * 255

        binary_roi = cv2.resize(
            binary_map, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST
        )
        # ======DEBUG======
        if self.debug_dir:
            cv2.imwrite(
                os.path.join(self.debug_dir, "stage1_localization", "binary_roi.png"),
                binary_roi,
            )
        # ======DEBUG======

        # Horizontally-oriented kernel to bridge scoreboard gaps (Resolution-Agile)
        kw = max(8, int(round(w_roi * 0.04)))
        kh = max(4, int(round(h_roi * 0.05)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
        closed_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_CLOSE, kernel)
        # ======DEBUG======
        if self.debug_dir:
            cv2.imwrite(
                os.path.join(self.debug_dir, "stage1_localization", "closed_roi.png"),
                closed_roi,
            )
        # ======DEBUG======

        contours, _ = cv2.findContours(
            closed_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # For debug candidate visualization
        vis_candidates = mid_frame.copy()

        candidates = []
        for idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h

            # Aspect ratio check
            aspect_pass = (
                self.config["score_aspect_min"]
                <= aspect_ratio
                <= self.config["score_aspect_max"]
            )
            # Minimum width check
            width_pass = w >= (0.05 * self.width)

            # Compute mean score in this candidate box
            rx1, rx2 = x // b, (x + w) // b
            ry1, ry2 = y // b, (y + h) // b
            mean_score_val = (
                score_map[ry1:ry2, rx1:rx2].mean() if rx2 > rx1 and ry2 > ry1 else 0
            )

            if aspect_pass and width_pass:
                candidates.append(((x, y, w, h), mean_score_val))
                color = (0, 255, 0)  # Green for passed candidates
            else:
                color = (0, 0, 255)  # Red for failed candidates

            # ======DEBUG======
            if self.debug_dir:
                cv2.rectangle(vis_candidates, (x, y), (x + w, y + h), color, 2)
                label = f"#{idx}: AR={aspect_ratio:.1f}, W={w}, S={mean_score_val:.2f}"
                cv2.putText(
                    vis_candidates,
                    label,
                    (x, max(y - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )
            # ======DEBUG======

        # ======DEBUG======
        if self.debug_dir:
            cv2.imwrite(
                os.path.join(
                    self.debug_dir, "stage1_localization", "all_candidates.png"
                ),
                vis_candidates,
            )
        # ======DEBUG======

        if not candidates:
            logger.warning(
                "Scoreboard localisation found no candidates! Falling back to default top-left box."
            )
            default_box = (
                int(0.02 * self.width),
                int(0.02 * self.height),
                int(0.18 * self.width),
                int(0.06 * self.height),
            )
            # ======DEBUG======
            if self.debug_dir:
                draw_and_save_bbox(
                    mid_frame,
                    default_box,
                    os.path.join(self.output_dir, "scoreboard_detected.png"),
                    "Scoreboard (Fallback)",
                )
                draw_and_save_bbox(
                    mid_frame,
                    default_box,
                    os.path.join(
                        self.debug_dir,
                        "stage1_localization",
                        "scoreboard_localized.png",
                    ),
                    "Scoreboard (Fallback)",
                )
            # ======DEBUG======
            return default_box

        # Sort candidates by score descending to prefer the region with best spatial gradient/low variance
        candidates.sort(key=lambda item: item[1], reverse=True)
        best_box, score_val = candidates[0]

        # Calibration Confidence Metric
        mask = np.zeros_like(score_map, dtype=bool)
        bx, by, bw, bh = best_box
        rx1, rx2 = bx // b, (bx + bw) // b
        ry1, ry2 = by // b, (by + bh) // b
        mask[ry1:ry2, rx1:rx2] = True

        mean_score_scoreboard = score_map[mask].mean() if np.any(mask) else 0
        mean_score_rest = score_map[~mask].mean() if np.any(~mask) else 1e-5
        confidence = mean_score_scoreboard / (mean_score_rest + 1e-5)

        logger.info(
            f"Scoreboard localised at {best_box} with score {score_val:.2f} and confidence {confidence:.2f}"
        )

        if confidence < self.config["calibration_confidence_min"]:
            logger.warning(
                f"Calibration confidence {confidence:.2f} is below threshold {self.config['calibration_confidence_min']:.2f}!"
            )

        # ======DEBUG======
        if self.debug_dir:
            draw_and_save_bbox(
                mid_frame,
                best_box,
                os.path.join(self.output_dir, "scoreboard_detected.png"),
                f"Scoreboard (Conf: {confidence:.2f})",
            )
            draw_and_save_bbox(
                mid_frame,
                best_box,
                os.path.join(
                    self.debug_dir, "stage1_localization", "scoreboard_localized.png"
                ),
                f"Scoreboard (Conf: {confidence:.2f})",
            )
        # ======DEBUG======

        return best_box

    def _isolate_score_region(self, clean_frames):
        """
        Isolates timer and score regions inside the scoreboard crop.

        New logic:
        1) OCR multiple clean frames -> consensus tokens
        2) Find timer using colon-first logic:
        - direct '12:34'
        - assembled '12' + ':' + '34'
        - fallback compact '1234'
        3) Remove timer region and collect remaining digit tokens as score candidates
        4) Infer split/unified score layout from those score candidates
        5) Refine/pad the resulting score boxes
        """
        bx, by, bw, bh = self.scoreboard_bbox

        # ---------------------------------------------------------
        # 1) Temporal variance map inside scoreboard crop (debug + fallback aid)
        # ---------------------------------------------------------
        crop_grays = []
        for frame in clean_frames:
            crop = frame[by : by + bh, bx : bx + bw]
            crop_grays.append(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))

        crop_grays = np.stack(crop_grays, axis=0)
        crop_variance = np.var(crop_grays, axis=0)

        # ======DEBUG======
        if self.debug_dir:
            plt.figure(figsize=(6, 3))
            plt.imshow(crop_variance, cmap="hot")
            plt.colorbar(label="Variance")
            plt.title("Scoreboard Crop Temporal Variance")
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_dir, "scoreboard_crop_variance.png"), dpi=150
            )
            plt.close()

            plt.figure(figsize=(6, 3))
            plt.imshow(crop_variance, cmap="hot")
            plt.colorbar(label="Variance")
            plt.title("Scoreboard Crop Temporal Variance")
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.debug_dir, "stage2_isolation", "crop_variance.png"),
                dpi=150,
            )
            plt.close()
        # ======DEBUG======

        # ---------------------------------------------------------
        # 2) OCR voting across multiple clean frames -> consensus tokens
        # ---------------------------------------------------------
        def compute_iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
            yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            areaA = boxA[2] * boxA[3]
            areaB = boxB[2] * boxB[3]
            unionArea = areaA + areaB - interArea
            return interArea / unionArea if unionArea > 0 else 0

        # Keep the existing digit cleaner; it is still useful later
        def clean_digit_like(text_val):
            if any(char in text_val for char in ("(", ")", "[", "]", "{", "}")):
                return None
            cleaned = re.sub(r"[^a-zA-Z0-9]", "", text_val)
            if not cleaned:
                return None
            mapping = {
                "o": "0",
                "O": "0",
                "i": "1",
                "I": "1",
                "l": "1",
                "s": "5",
                "S": "5",
                "b": "6",
                "B": "8",
                "g": "9",
                "q": "9",
                "z": "2",
                "Z": "2",
                "t": "7",
                "T": "7",
            }
            mapped = "".join([mapping.get(c, c) for c in cleaned])
            if mapped.isdigit() and 1 <= len(mapped) <= 2:
                return mapped
            return None

        # Find up to 5 valid calibration frames by sweeping sub-intervals with verification logging
        k_samples = 5
        calib_pool = []
        interval_len = len(clean_frames) // k_samples if len(clean_frames) >= k_samples else 1

        for i in range(k_samples):
            sub_start = i * interval_len
            sub_end = sub_start + interval_len if i < k_samples - 1 else len(clean_frames)
            if sub_start >= len(clean_frames):
                break

            found = False
            for idx in range(sub_start, sub_end):
                frame = clean_frames[idx]
                if self._validate_frame_relaxed(frame):
                    logger.info(f"Verified frame index {idx} as eligible calibration frame.")
                    calib_pool.append((idx, frame))
                    found = True
                    break
                else:
                    logger.warning(f"Frame index {idx} failed relaxed calibration check (no timer or missing score digits).")

            if not found:
                logger.warning(f"Could not find verified calibration frame in sub-interval #{i+1} [{sub_start}, {sub_end}]. Falling back to frame index {sub_start}.")
                calib_pool.append((sub_start, clean_frames[sub_start]))

        k_samples = len(calib_pool)
        sample_indices = list(range(k_samples))

        all_raw_tokens = []
        for i in sample_indices:
            f_idx, frame = calib_pool[i]
            scoreboard_crop = frame[by : by + bh, bx : bx + bw]
            try:
                res_list = list(self._ocr_pipeline.predict(scoreboard_crop))
                if res_list and len(res_list) > 0:
                    res = res_list[0]
                    texts = res.get("rec_texts", [])
                    scores = res.get("rec_scores", [])
                    polys = res.get("dt_polys", [])
                    for i in range(len(texts)):
                        text = texts[i].strip()
                        if not text:
                            continue
                        conf = scores[i] * 100.0
                        poly = polys[i]
                        xmin = float(np.min(poly[:, 0]))
                        xmax = float(np.max(poly[:, 0]))
                        ymin = float(np.min(poly[:, 1]))
                        ymax = float(np.max(poly[:, 1]))
                        all_raw_tokens.append(
                            {
                                "frame_idx": int(f_idx),
                                "text": text,
                                "bbox": (xmin, ymin, xmax - xmin, ymax - ymin),
                                "conf": conf,
                            }
                        )
            except Exception as e:
                logger.error(
                    f"Error running PaddleOCR in calibration on frame index {f_idx}: {e}"
                )

        frame_tokens = defaultdict(list)

        for tok in all_raw_tokens:
            frame_tokens[tok["frame_idx"]].append(tok)

        score_initialized = False
        self.composite_initialized = False
        score_found = False

        # Pass 1: Look for exactly 2 standalone digit-only tokens (matches original behavior, regression-free)
        for frame_idx in sorted(frame_tokens.keys()):
            tokens = frame_tokens[frame_idx]
            zeros = [
                t for t in tokens if self._normalize_ocr_token(t["text"]).isdigit()
            ]
            if len(zeros) != 2:
                continue
            zeros.sort(key=lambda t: t["bbox"][0])
            home = zeros[0]
            away = zeros[1]
            score_found = True
            break

        # Pass 2: Fallback for composite scoreboards (like Premier League)
        if not score_found:
            for frame_idx in sorted(frame_tokens.keys()):
                tokens = frame_tokens[frame_idx]
                digit_tokens = []
                for t in tokens:
                    text = t["text"]
                    # Skip timer and aggregate score patterns
                    if any(delim in text for delim in (":", ";", "(", ")", "[", "]")):
                        continue
                    if len(text.strip()) == 1 and text.strip() in ("O", "o", "0"):
                        normalized = "0"
                    else:
                        normalized = text
                    tx, ty, tw, th = t["bbox"]
                    n_chars = len(normalized)
                    if n_chars == 0:
                        continue
                    for match in re.finditer(r"\d", normalized):
                        digit_char = match.group()
                        idx = match.start()
                        char_w = tw / n_chars
                        char_x = tx + (idx * char_w)
                        digit_tokens.append({
                            "text": digit_char,
                            "bbox": (int(char_x), int(ty), int(char_w), int(th)),
                            "frame_idx": t["frame_idx"],
                            "conf": t.get("conf", 1.0),
                        })

                zeros = digit_tokens
                if len(zeros) != 2:
                    continue

                zeros.sort(key=lambda t: t["bbox"][0])
                home = zeros[0]
                away = zeros[1]
                score_found = True
                self.composite_initialized = True
                break

        if score_found:
            hx, hy, hw, hh = home["bbox"]
            ax, ay, aw, ah = away["bbox"]

            self.home_score_bbox_rel = (
                int(hx),
                int(hy),
                int(hw),
                int(hh),
            )

            self.away_score_bbox_rel = (
                int(ax),
                int(ay),
                int(aw),
                int(ah),
            )

            self.is_split_score = True
            score_initialized = True

            logger.info("Initialized score boxes from 0-0 frame")
            if self.composite_initialized:
                logger.info(
                    "Composite scoreboard fallback was initialized. "
                    "Lowering min_ocr_confidence to 50.0 to handle merged team-score crops."
                )
                self.config["min_ocr_confidence"] = 50.0
        # pre-filter low-confidence tokens before clustering
        all_raw_tokens = [t for t in all_raw_tokens if t["conf"] >= 45]

        # cluster by IoU to form consensus tokens
        clusters = []
        for tok in all_raw_tokens:
            placed = False
            for cluster in clusters:
                if compute_iou(tok["bbox"], cluster["anchor_bbox"]) > 0.4:
                    cluster["tokens"].append(tok)
                    placed = True
                    break
            if not placed:
                clusters.append({"anchor_bbox": tok["bbox"], "tokens": [tok]})

        min_majority = max(2, int(round(k_samples / 2.0)))
        all_tokens = []
        for cluster in clusters:
            tokens = cluster["tokens"]
            unique_frames = len(set(t["frame_idx"] for t in tokens))
            if unique_frames >= min_majority:
                best = max(tokens, key=lambda t: t["conf"])
                mean_conf = sum(t["conf"] for t in tokens) / len(tokens)
                all_tokens.append(
                    {
                        "text": best["text"],
                        "bbox": cluster["anchor_bbox"],
                        "conf": mean_conf,
                    }
                )

        # Pass 3: Look for exactly 2 standalone digit tokens in consensus tokens (all_tokens)
        if not score_found:
            zeros = [
                t for t in all_tokens if self._normalize_ocr_token(t["text"]).isdigit()
            ]
            if len(zeros) == 2:
                zeros.sort(key=lambda t: t["bbox"][0])
                home = zeros[0]
                away = zeros[1]
                score_found = True
                logger.info("Initialized score boxes from consensus standalone tokens")

        # Pass 4: Look for exactly 2 composite digit tokens in consensus tokens (all_tokens)
        if not score_found:
            digit_tokens = []
            for t in all_tokens:
                text = t["text"]
                if any(delim in text for delim in (":", ";", "(", ")", "[", "]")):
                    continue
                if len(text.strip()) == 1 and text.strip() in ("O", "o", "0"):
                    normalized = "0"
                else:
                    normalized = text
                tx, ty, tw, th = t["bbox"]
                n_chars = len(normalized)
                if n_chars == 0:
                    continue
                for match in re.finditer(r"\d", normalized):
                    digit_char = match.group()
                    idx = match.start()
                    char_w = tw / n_chars
                    char_x = tx + (idx * char_w)
                    digit_tokens.append({
                        "text": digit_char,
                        "bbox": (int(char_x), int(ty), int(char_w), int(th)),
                        "conf": t.get("conf", 1.0),
                    })

            zeros = digit_tokens
            if len(zeros) == 2:
                zeros.sort(key=lambda t: t["bbox"][0])
                home = zeros[0]
                away = zeros[1]
                score_found = True
                self.composite_initialized = True
                logger.info("Initialized score boxes from consensus composite tokens")

        if score_found and not score_initialized:
            hx, hy, hw, hh = home["bbox"]
            ax, ay, aw, ah = away["bbox"]

            self.home_score_bbox_rel = (
                int(hx),
                int(hy),
                int(hw),
                int(hh),
            )

            self.away_score_bbox_rel = (
                int(ax),
                int(ay),
                int(aw),
                int(ah),
            )

            self.is_split_score = True
            score_initialized = True
            if self.composite_initialized:
                logger.info(
                    "Composite scoreboard fallback was initialized from consensus tokens. "
                    "Lowering min_ocr_confidence to 50.0 to handle merged team-score crops."
                )
                self.config["min_ocr_confidence"] = 50.0

        # ---------------------------------------------------------
        # 3) Debug visualization for OCR tokens
        # ---------------------------------------------------------
        scoreboard_crop_vis = None
        if self.debug_dir and len(sample_indices) > 0:
            sample_frame = clean_frames[sample_indices[0]]
            scoreboard_crop_vis = sample_frame[by : by + bh, bx : bx + bw].copy()

            raw_vis = scoreboard_crop_vis.copy()
            for tok in all_raw_tokens:
                tx, ty, tw, th = [int(c) for c in tok["bbox"]]
                cv2.rectangle(raw_vis, (tx, ty), (tx + tw, ty + th), (255, 0, 0), 1)
                label = f"{tok['text']} ({tok['conf']:.0f}%)"
                cv2.putText(
                    raw_vis,
                    label,
                    (tx, max(ty - 2, 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 0, 0),
                    1,
                )
            cv2.imwrite(
                os.path.join(self.debug_dir, "stage2_isolation", "ocr_raw_tokens.png"),
                raw_vis,
            )

            con_vis = scoreboard_crop_vis.copy()
            for tok in all_tokens:
                tx, ty, tw, th = [int(c) for c in tok["bbox"]]
                cv2.rectangle(con_vis, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 1)
                label = f"{tok['text']} ({tok['conf']:.0f}%)"
                cv2.putText(
                    con_vis,
                    label,
                    (tx, max(ty - 2, 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 255, 0),
                    1,
                )
            cv2.imwrite(
                os.path.join(
                    self.debug_dir, "stage2_isolation", "ocr_consensus_tokens.png"
                ),
                con_vis,
            )

        # ---------------------------------------------------------
        # 4) TIMER DETECTION (new colon-first logic)
        # ---------------------------------------------------------
        timer_candidate = self._find_timer_from_tokens(all_tokens, bw, bh)

        if timer_candidate is not None:
            self.timer_bbox_rel = self._pad_rel_box(
                timer_candidate["bbox"], bw, bh, pad=2
            )
            logger.info(
                f"Timer found via colon-first logic: {self.timer_bbox_rel} | "
                f"text='{timer_candidate['text']}' | mode={timer_candidate['kind']}"
            )
        else:
            self.timer_bbox_rel = None
            logger.warning(
                "Timer not found from OCR tokens. Falling back to variance-based timer localization..."
            )

            # Fallback: variance over the WHOLE scoreboard crop (not left half)
            var_max = crop_variance.max()
            if var_max > 0:
                normalized_var = (crop_variance / var_max * 255).astype(np.uint8)
                _, var_thresh = cv2.threshold(
                    normalized_var, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                contours, _ = cv2.findContours(
                    var_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                # Prefer components with timer-like size and aspect
                best_box = None
                best_area = -1
                for contour in contours:
                    tx, ty, tw, th = cv2.boundingRect(contour)
                    if tw < max(8, int(0.05 * bw)):
                        continue
                    if th < max(8, int(0.20 * bh)):
                        continue
                    if tw > int(0.50 * bw):
                        continue
                    if th > int(0.90 * bh):
                        continue
                    area = tw * th
                    if area > best_area:
                        best_area = area
                        best_box = (tx, ty, tw, th)

                if best_box is not None:
                    self.timer_bbox_rel = self._pad_rel_box(best_box, bw, bh, pad=2)
                    logger.info(
                        f"Timer fallback via full-scoreboard variance: {self.timer_bbox_rel}"
                    )
                else:
                    logger.warning(
                        "Variance fallback failed; leaving timer_bbox_rel=None"
                    )

        if self.timer_bbox_rel is not None:
            tx, ty, tw, th = self.timer_bbox_rel
            self.timer_bbox_abs = (int(bx + tx), int(by + ty), int(tw), int(th))
        else:
            self.timer_bbox_abs = None

        # ---------------------------------------------------------
        # 5) SCORE TOKENS = all digit tokens excluding timer region
        # ---------------------------------------------------------
        score_tokens = self._find_score_tokens_excluding_timer(
            all_tokens, self.timer_bbox_rel, bw, bh
        )

        logger.info(
            f"Score candidate tokens (excluding timer): "
            f"{[(self._normalize_ocr_token(t['text']), t['bbox']) for t in score_tokens]}"
        )

        # Extra conservative filter:
        # remove tokens that are too close to timer if timer exists and there are many candidates
        if self.timer_bbox_rel is not None and len(score_tokens) > 2:
            tx, ty, tw, th = self.timer_bbox_rel
            timer_cx = tx + tw / 2.0
            filtered = []
            for tok in score_tokens:
                bx2 = tok["bbox"]
                tok_cx = bx2[0] + bx2[2] / 2.0
                # keep tokens sufficiently separated from timer center OR tokens with high confidence
                if abs(tok_cx - timer_cx) > 0.12 * bw or tok["conf"] >= 90:
                    filtered.append(tok)
            if len(filtered) >= 1:
                score_tokens = filtered

        # ---------------------------------------------------------
        # 6) Infer score layout (split or unified)
        # ---------------------------------------------------------
        if not score_initialized:
            layout = self._infer_score_layout_from_tokens(score_tokens, bw, bh)

            if layout is None:
                logger.warning(
                    "No score tokens found after excluding timer. Falling back to default unified score priors."
                )
                self.is_split_score = False
                self.score_bbox_rel = (
                    int(0.40 * bw),
                    int(0.25 * bh),
                    int(0.18 * bw),
                    int(0.40 * bh),
                )
            else:
                if layout["mode"] == "split":
                    self.is_split_score = True
                    self.home_score_bbox_rel = layout["home_bbox_rel"]
                    self.away_score_bbox_rel = layout["away_bbox_rel"]
                    logger.info(
                        f"Detected Split Scoreboard from score tokens. "
                        f"Home Rel: {self.home_score_bbox_rel}, Away Rel: {self.away_score_bbox_rel}"
                    )
                else:
                    self.is_split_score = False
                    self.score_bbox_rel = layout["score_bbox_rel"]
                    logger.info(
                        f"Detected Unified Scoreboard from score tokens: {self.score_bbox_rel}"
                    )

        # ---------------------------------------------------------
        # 7) Debug visualization of timer + score token layout
        # ---------------------------------------------------------
        mid_frame = clean_frames[len(clean_frames) // 2]
        scoreboard_crop = mid_frame[by : by + bh, bx : bx + bw]

        if self.debug_dir:
            layout_vis = scoreboard_crop.copy()

            if self.timer_bbox_rel is not None:
                tx, ty, tw, th = self.timer_bbox_rel
                cv2.rectangle(
                    layout_vis,
                    (int(tx), int(ty)),
                    (int(tx + tw), int(ty + th)),
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    layout_vis,
                    "Timer",
                    (int(tx), int(max(ty - 5, 12))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                )

            for tok in score_tokens:
                tx, ty, tw, th = [int(c) for c in tok["bbox"]]
                txt = self._normalize_ocr_token(tok["text"])
                cv2.rectangle(layout_vis, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 1)
                cv2.putText(
                    layout_vis,
                    txt,
                    (tx, max(ty - 2, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 255, 0),
                    1,
                )

            cv2.imwrite(
                os.path.join(
                    self.debug_dir, "stage2_isolation", "layout_candidates.png"
                ),
                layout_vis,
            )

        # ---------------------------------------------------------
        # 8) Refine and pad score boxes
        # ---------------------------------------------------------
        if self.is_split_score:
            refined_boxes = {}
            for label, bbox in [
                ("Home", self.home_score_bbox_rel),
                ("Away", self.away_score_bbox_rel),
            ]:
                if self.composite_initialized:
                    refined_boxes[label] = bbox
                    continue
                rx, ry, rw, rh = [int(c) for c in bbox]
                subcrop = scoreboard_crop[ry : ry + rh, rx : rx + rw]

                scale = 3
                subcrop_up = cv2.resize(
                    subcrop,
                    (
                        max(1, subcrop.shape[1] * scale),
                        max(1, subcrop.shape[0] * scale),
                    ),
                    interpolation=cv2.INTER_CUBIC,
                )
                gray_up = cv2.cvtColor(subcrop_up, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(
                    gray_up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                contours, _ = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                if self.debug_dir:
                    cv2.imwrite(
                        os.path.join(
                            self.debug_dir,
                            "stage2_isolation",
                            f"refine_{label}_subcrop_up.png",
                        ),
                        subcrop_up,
                    )
                    cv2.imwrite(
                        os.path.join(
                            self.debug_dir,
                            "stage2_isolation",
                            f"refine_{label}_thresh.png",
                        ),
                        thresh,
                    )

                min_h = int(0.15 * max(1, subcrop.shape[0])) * scale
                max_h = int(0.95 * max(1, subcrop.shape[0])) * scale
                min_w = int(0.10 * max(1, subcrop.shape[1])) * scale
                max_w = int(0.95 * max(1, subcrop.shape[1])) * scale

                best_bbox = None
                best_area = -1
                for contour in contours:
                    cx, cy, cw, ch = cv2.boundingRect(contour)
                    area = cw * ch
                    if min_h <= ch <= max_h and min_w <= cw <= max_w:
                        if area > best_area:
                            best_area = area
                            best_bbox = (cx, cy, cw, ch)

                if best_bbox:
                    cx, cy, cw, ch = [c // scale for c in best_bbox]
                    pad_w = max(2, int(cw * 0.08))
                    pad_h = max(2, int(ch * 0.08))
                    ref_x = max(0, rx + cx - pad_w)
                    ref_y = max(0, ry + cy - pad_h)
                    ref_w = min(bw - ref_x, cw + 2 * pad_w)
                    ref_h = min(bh - ref_y, ch + 2 * pad_h)
                    refined_boxes[label] = (ref_x, ref_y, ref_w, ref_h)
                    logger.info(
                        f"Refined {label} score bbox rel to {refined_boxes[label]} (original: {bbox})"
                    )
                else:
                    # pad_w = max(4, int(rw * 0.15))
                    # pad_h = max(4, int(rh * 0.15))
                    # ref_x = max(0, rx - pad_w)
                    # ref_y = max(0, ry - pad_h)
                    # ref_w = min(bw - ref_x, rw + 2 * pad_w)
                    # ref_h = min(bh - ref_y, rh + 2 * pad_h)
                    # refined_boxes[label] = (ref_x, ref_y, ref_w, ref_h)
                    # logger.info(f"Refinement failed for {label}, using padded original {refined_boxes[label]} (original: {bbox})")
                    refined_boxes[label] = (rx, ry, rw, rh)
                    logger.info(f"Refinement failed for {label}, using original {bbox}")

            self.home_score_bbox_rel = refined_boxes["Home"]
            self.away_score_bbox_rel = refined_boxes["Away"]

            hx, hy, hw, hh = self.home_score_bbox_rel
            ax, ay, aw, ah = self.away_score_bbox_rel

            self.home_score_bbox_abs = (int(bx + hx), int(by + hy), int(hw), int(hh))
            self.away_score_bbox_abs = (int(bx + ax), int(by + ay), int(aw), int(ah))

        else:
            sx, sy, sw, sh = [int(c) for c in self.score_bbox_rel]
            score_x_abs = bx + sx
            score_y_abs = by + sy

            pad_w = max(2, int(sw * 0.08))
            pad_h = max(2, int(sh * 0.08))

            pad_x = int(max(0, score_x_abs - pad_w))
            pad_y = int(max(0, score_y_abs - pad_h))
            pad_w_final = int(min(self.width - pad_x, sw + 2 * pad_w))
            h_pad_final = int(min(self.height - pad_y, sh + 2 * pad_h))
            self.score_bbox_abs = (pad_x, pad_y, pad_w_final, h_pad_final)

        # ---------------------------------------------------------
        # 9) Save visualization of isolated timer/score boxes
        # ---------------------------------------------------------
        if self.is_split_score:
            vis = scoreboard_crop.copy()

            hx, hy, hw, hh = self.home_score_bbox_rel
            cv2.rectangle(
                vis, (int(hx), int(hy)), (int(hx + hw), int(hy + hh)), (0, 255, 0), 2
            )
            cv2.putText(
                vis,
                "Home Score",
                (int(hx), int(max(hy - 5, 12))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )

            ax, ay, aw, ah = self.away_score_bbox_rel
            cv2.rectangle(
                vis, (int(ax), int(ay)), (int(ax + aw), int(ay + ah)), (0, 255, 0), 2
            )
            cv2.putText(
                vis,
                "Away Score",
                (int(ax), int(max(ay - 5, 12))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )

            if self.timer_bbox_rel:
                tx, ty, tw, th = self.timer_bbox_rel
                cv2.rectangle(
                    vis,
                    (int(tx), int(ty)),
                    (int(tx + tw), int(ty + th)),
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    vis,
                    "Timer",
                    (int(tx), int(max(ty - 5, 12))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                )

            # ======DEBUG======
            if self.debug_dir:
                cv2.imwrite(os.path.join(self.output_dir, "score_isolated.png"), vis)
                cv2.imwrite(
                    os.path.join(
                        self.debug_dir, "stage2_isolation", "score_isolated.png"
                    ),
                    vis,
                )
            # ======DEBUG======

        else:
            # ======DEBUG======
            if self.debug_dir:
                save_isolated_subcrops(
                    scoreboard_crop,
                    self.score_bbox_rel,
                    self.timer_bbox_rel,
                    os.path.join(self.output_dir, "score_isolated.png"),
                )
                save_isolated_subcrops(
                    scoreboard_crop,
                    self.score_bbox_rel,
                    self.timer_bbox_rel,
                    os.path.join(
                        self.debug_dir, "stage2_isolation", "score_isolated.png"
                    ),
                )
            # ======DEBUG======

    def _create_scoreboard_template(self, clean_frames):
        """Creates a static background template and mask from clean frames."""
        logger.info("Creating scoreboard template and static mask...")
        bx, by, bw, bh = self.scoreboard_bbox

        # 1. Collect grayscale scoreboard crops
        gray_crops = []
        for frame in clean_frames:
            crop = frame[by : by + bh, bx : bx + bw]
            gray_crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))

        # 2. Compute median scoreboard image template
        self.scoreboard_template = np.median(gray_crops, axis=0).astype(np.uint8)

        # 3. Create binary mask (1 for static, 0 for dynamic regions)
        mask = np.ones((bh, bw), dtype=np.uint8)

        # Mask out dynamic timer region
        if self.timer_bbox_rel is not None:
            tx, ty, tw, th = [int(c) for c in self.timer_bbox_rel]
            mask[ty : ty + th, tx : tx + tw] = 0

        # Mask out dynamic score regions
        if self.is_split_score:
            if self.home_score_bbox_rel is not None:
                hx, hy, hw, hh = [int(c) for c in self.home_score_bbox_rel]
                mask[hy : hy + hh, hx : hx + hw] = 0
            if self.away_score_bbox_rel is not None:
                ax, ay, aw, ah = [int(c) for c in self.away_score_bbox_rel]
                mask[ay : ay + ah, ax : ax + aw] = 0
        else:
            if self.score_bbox_rel is not None:
                sx, sy, sw, sh = [int(c) for c in self.score_bbox_rel]
                mask[sy : sy + sh, sx : sx + sw] = 0

        self.scoreboard_mask = mask

        # 4. Compute MAD standard deviation on calibration frames to define threshold
        mads = []
        temp_blurred = cv2.GaussianBlur(self.scoreboard_template, (3, 3), 0)
        for g_crop in gray_crops:
            crop_blurred = cv2.GaussianBlur(g_crop, (3, 3), 0)
            diff = cv2.absdiff(crop_blurred, temp_blurred)
            # Compute mean absolute difference only in the masked (static) area
            masked_diff = diff[self.scoreboard_mask == 1]
            if len(masked_diff) > 0:
                mad = float(np.mean(masked_diff))
                mads.append(mad)

        if mads:
            # Add a 30% safety margin to prevent false negative absent classifications on noisier validation/tracking frames
            self.scoreboard_threshold = float(np.percentile(mads, 99)) * 1.30
            logger.info(
                f"Scoreboard template comparison threshold (99th percentile with 30% margin): {self.scoreboard_threshold:.4f} "
                f"(min MAD: {np.min(mads):.4f}, max MAD: {np.max(mads):.4f})"
            )
        else:
            # Fallback if no frames
            self.scoreboard_threshold = 15.0
            logger.warning(
                "Could not calculate MAD statistics; using fallback threshold of 15.0"
            )

        # ======DEBUG======
        if self.debug_dir:
            # Save template and mask for verification
            cv2.imwrite(
                os.path.join(self.output_dir, "scoreboard_template.png"),
                self.scoreboard_template,
            )
            cv2.imwrite(
                os.path.join(self.output_dir, "scoreboard_mask.png"),
                self.scoreboard_mask * 255,
            )
            logger.info("Saved scoreboard template and mask debug visualizations.")
        # ======DEBUG======

    def _perform_ocr(self, crop1, crop2=None, sec=None, min_conf=None):
        """Runs PaddleOCR on the score crop(s) to predict the score."""
        min_c = min_conf if min_conf is not None else self.config["min_ocr_confidence"]
        if crop2 is None:
            # Unified mode
            # 1. Change detection check
            if self.config.get("use_change_cache", True):
                if self.last_score_crop is not None and self.last_reading is not None:
                    diff = np.mean(cv2.absdiff(crop1, self.last_score_crop))
                    if diff < 3.5:
                        self.ocr_skip_count += 1
                        return self.last_reading

            self.ocr_run_count += 1

            if len(crop1.shape) == 2 or crop1.shape[2] == 1:
                crop1_color = cv2.cvtColor(crop1, cv2.COLOR_GRAY2BGR)
            else:
                crop1_color = crop1.copy()

            # Upscale and pad the crop for better recognition
            h_im, w_im = crop1_color.shape[:2]
            scale_im = 3
            crop1_color = cv2.resize(
                crop1_color,
                (w_im * scale_im, h_im * scale_im),
                interpolation=cv2.INTER_CUBIC,
            )
            crop1_color = cv2.copyMakeBorder(
                crop1_color, 4, 4, 4, 4, cv2.BORDER_REPLICATE
            )

            best_val = None
            best_conf = -1
            try:
                res_list = list(self._ocr_rec.predict(crop1_color))
                if res_list and len(res_list) > 0:
                    res = res_list[0]
                    text = res["rec_text"].strip()
                    conf = float(res["rec_score"]) * 100.0

                    cleaned_no_space = re.sub(r"[^0-9]", "", text).strip()

                    if len(cleaned_no_space) == 2 and cleaned_no_space.isdigit():
                        home = int(cleaned_no_space[0])
                        away = int(cleaned_no_space[1])
                        best_val = (home, away)
                        best_conf = conf
                    elif len(cleaned_no_space) == 3 and cleaned_no_space.isdigit():
                        val = int(cleaned_no_space)
                        if self.last_accepted_score:
                            lh, la = self.last_accepted_score
                            split1 = (val // 10, val % 10)
                            split2 = (val // 100, val % 100)
                            if (
                                split1[0] >= lh
                                and split1[1] >= la
                                and (split1[0] - lh + split1[1] - la) <= 2
                            ):
                                best_val = split1
                                best_conf = conf
                            elif (
                                split2[0] >= lh
                                and split2[1] >= la
                                and (split2[0] - lh + split2[1] - la) <= 2
                            ):
                                best_val = split2
                                best_conf = conf
                            else:
                                best_val = split1
                                best_conf = conf
                        else:
                            best_val = (val // 10, val % 10)
                            best_conf = conf
            except Exception as e:
                logger.error(f"Error in unified mode PaddleOCR: {e}")

            if best_val is not None and best_conf >= min_c:
                self.last_reading = best_val
                self.last_score_crop = crop1.copy()

                if sec is not None and self.debug_dir is not None:
                    hit_dir = os.path.join(
                        self.debug_dir, "ocr_hits", f"sec_{sec:.3f}_unified"
                    )
                    os.makedirs(hit_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(hit_dir, "crop_raw.png"), crop1)
                    cv2.imwrite(os.path.join(hit_dir, "crop_prep.png"), crop1_color)
                    result = {
                        "prediction": f"{best_val[0]}-{best_val[1]}",
                        "confidence": best_conf,
                        "mode": "unified",
                        "preprocessing_method": "paddle_rec",
                        "timestamp_sec": sec,
                    }
                    with open(os.path.join(hit_dir, "result.json"), "w") as f:
                        json.dump(result, f, indent=4)
                return best_val
            return None
        else:
            # Split mode: crop1=home_crop_raw, crop2=away_crop_raw
            # 1. Change detection check
            if self.config.get("use_change_cache", True):
                if (
                    self.last_home_crop is not None
                    and self.last_away_crop is not None
                    and self.last_reading is not None
                ):
                    home_diff = np.mean(cv2.absdiff(crop1, self.last_home_crop))
                    away_diff = np.mean(cv2.absdiff(crop2, self.last_away_crop))
                    if home_diff < 3.5 and away_diff < 3.5:
                        self.ocr_skip_count += 1
                        return self.last_reading

            self.ocr_run_count += 1

            def prepare_crop(c):
                if len(c.shape) == 2 or c.shape[2] == 1:
                    c_color = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
                else:
                    c_color = c.copy()

                # Upscale and pad the crop for better recognition
                h_im, w_im = c_color.shape[:2]
                scale_im = 3
                c_color = cv2.resize(
                    c_color,
                    (w_im * scale_im, h_im * scale_im),
                    interpolation=cv2.INTER_CUBIC,
                )
                c_color = cv2.copyMakeBorder(c_color, 4, 4, 4, 4, cv2.BORDER_REPLICATE)
                return c_color

            home_color = prepare_crop(crop1)
            away_color = prepare_crop(crop2)

            home_val, home_conf = None, -1
            away_val, away_conf = None, -1

            def ocr_single(c):
                try:
                    res_list = list(self._ocr_rec.predict(c))
                    if res_list and len(res_list) > 0:
                        res = res_list[0]
                        text = res["rec_text"].strip()
                        conf = float(res["rec_score"]) * 100.0
                        text_clean = re.sub(r"[^0-9]", "", text).strip()
                        if text_clean.isdigit():
                            return int(text_clean), conf
                except Exception as e:
                    logger.error(f"Error in split mode PaddleOCR: {e}")
                return None, -1

            home_val, home_conf = ocr_single(home_color)
            away_val, away_conf = ocr_single(away_color)

            if home_val is not None and home_conf < min_c:
                home_val = None
            if away_val is not None and away_conf < min_c:
                away_val = None

            # Save hits
            if sec is not None:
                if home_val is not None and self.debug_dir is not None:
                    hit_dir = os.path.join(
                        self.debug_dir, "ocr_hits", f"sec_{sec:.3f}_home"
                    )
                    os.makedirs(hit_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(hit_dir, "crop_raw.png"), crop1)
                    cv2.imwrite(os.path.join(hit_dir, "crop_prep.png"), home_color)
                    result = {
                        "prediction": home_val,
                        "confidence": home_conf,
                        "mode": "split_home",
                        "preprocessing_method": "paddle_rec",
                        "timestamp_sec": sec,
                    }
                    with open(os.path.join(hit_dir, "result.json"), "w") as f:
                        json.dump(result, f, indent=4)

                if away_val is not None and self.debug_dir is not None:
                    hit_dir = os.path.join(
                        self.debug_dir, "ocr_hits", f"sec_{sec:.3f}_away"
                    )
                    os.makedirs(hit_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(hit_dir, "crop_raw.png"), crop2)
                    cv2.imwrite(os.path.join(hit_dir, "crop_prep.png"), away_color)
                    result = {
                        "prediction": away_val,
                        "confidence": away_conf,
                        "mode": "split_away",
                        "preprocessing_method": "paddle_rec",
                        "timestamp_sec": sec,
                    }
                    with open(os.path.join(hit_dir, "result.json"), "w") as f:
                        json.dump(result, f, indent=4)

            if home_val is not None and away_val is not None:
                self.last_reading = (home_val, away_val)
                self.last_home_crop = crop1.copy()
                self.last_away_crop = crop2.copy()
                return self.last_reading
            return None

    def scan_scoreboard_presence(self, start_sec, end_sec):
        """Scans frames in [start_sec, end_sec] to find scoreboard absence periods.

        Returns a list of (abs_start_sec, abs_end_sec) tuples representing contiguous
        intervals where the scoreboard HUD is not visible.
        """
        scan_fps = self.config["scoreboard_scan_fps"]
        step = 1.0 / scan_fps
        min_dur = self.config["min_absence_duration"]

        logger.info(
            f"Scanning scoreboard presence from {start_sec:.1f}s to {end_sec:.1f}s "
            f"(scan_fps={scan_fps}, min_absence={min_dur:.1f}s)..."
        )

        with av.open(self.video_path) as container:
            video_stream = container.streams.video[0]
            time_base = float(video_stream.time_base)

            absent_readings = []
            timestamps = np.arange(start_sec, end_sec, step)

            for t in timestamps:
                t = float(t)
                pts = int(t / time_base)
                try:
                    container.seek(pts, stream=video_stream)
                    for frame in container.decode(video=0):
                        img = frame.to_ndarray(format="bgr24")
                        present = self._is_scoreboard_present(img)
                        if not present:
                            absent_readings.append(t)
                        break
                except Exception:
                    pass

        if not absent_readings:
            logger.info("Scoreboard presence scan: no absence periods detected")
            return []

        # Group consecutive absent readings into contiguous intervals
        absence_periods = []
        period_start = absent_readings[0]
        prev_t = absent_readings[0]

        for t in absent_readings[1:]:
            # If gap between readings is > 2x the step, treat as separate period
            if t - prev_t > step * 2.5:
                if prev_t - period_start >= min_dur:
                    absence_periods.append((period_start, prev_t))
                period_start = t
            prev_t = t

        # Close the last period
        if prev_t - period_start >= min_dur:
            absence_periods.append((period_start, prev_t))

        logger.info(
            f"Scoreboard presence scan: {len(absence_periods)} absence period(s) found "
            f"({len(absent_readings)} absent frames total)"
        )
        for i, (s, e) in enumerate(absence_periods):
            logger.info(f"  Absence #{i+1}: {s:.2f}s - {e:.2f}s (dur={e - s:.2f}s)")

        return absence_periods

    def _is_scoreboard_present(self, frame):
        """Checks if the scoreboard is visible in the given frame using template matching."""
        if (
            self.scoreboard_template is None
            or self.scoreboard_mask is None
            or self.scoreboard_threshold is None
        ):
            return True  # Fallback: proceed to OCR if template not initialized

        bx, by, bw, bh = self.scoreboard_bbox
        # Crop scoreboard region
        crop = frame[by : by + bh, bx : bx + bw]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Apply slight Gaussian blur to smooth out alignment/noise
        crop_blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        temp_blurred = cv2.GaussianBlur(self.scoreboard_template, (3, 3), 0)

        # Calculate Mean Absolute Difference on static mask area
        diff = cv2.absdiff(crop_blurred, temp_blurred)
        masked_diff = diff[self.scoreboard_mask == 1]

        if len(masked_diff) == 0:
            return True

        mad = float(np.mean(masked_diff))

        if mad > self.scoreboard_threshold:
            logger.debug(
                f"Scoreboard classified as ABSENT (MAD: {mad:.4f} > Threshold: {self.scoreboard_threshold:.4f})"
            )
            return False

        return True

    def _read_timer_from_frame(self, img):
        """OCR reads the game timer from a frame crop. Returns formatted 'MM:SS' string or None."""
        if self.timer_bbox_abs is None:
            return None

        tx, ty, tw, th = self.timer_bbox_abs
        h_img, w_img = img.shape[:2]
        tx_val = max(0, min(tx, w_img - 1))
        ty_val = max(0, min(ty, h_img - 1))
        tw_val = max(1, min(tw, w_img - tx_val))
        th_val = max(1, min(th, h_img - ty_val))

        timer_crop = img[ty_val : ty_val + th_val, tx_val : tx_val + tw_val]
        if timer_crop.size == 0:
            return None

        if len(timer_crop.shape) == 2 or timer_crop.shape[2] == 1:
            crop_color = cv2.cvtColor(timer_crop, cv2.COLOR_GRAY2BGR)
        else:
            crop_color = timer_crop.copy()

        h_im, w_im = crop_color.shape[:2]
        scale_im = 3
        crop_color = cv2.resize(
            crop_color,
            (w_im * scale_im, h_im * scale_im),
            interpolation=cv2.INTER_CUBIC,
        )
        crop_color = cv2.copyMakeBorder(
            crop_color, 4, 4, 4, 4, cv2.BORDER_REPLICATE
        )

        try:
            res_list = list(self._ocr_rec.predict(crop_color))
            if res_list and len(res_list) > 0:
                res = res_list[0]
                text = res["rec_text"].strip()
                norm = self._normalize_ocr_token(text)

                # 1. MM:SS
                match_colon = re.search(r"(\d{1,3})[\:\;\.\,]+(\d{2})", norm)
                if match_colon:
                    mins = int(match_colon.group(1))
                    secs = int(match_colon.group(2))
                    if 0 <= mins <= self.config["timer_max_minutes"] and 0 <= secs <= self.config["timer_max_seconds"]:
                        return f"{mins:02d}:{secs:02d}"

                # 2. Compact MMSS
                digits = re.sub(r"\D", "", norm)
                if len(digits) in (3, 4):
                    mins = int(digits[:-2])
                    secs = int(digits[-2:])
                    if 0 <= mins <= self.config["timer_max_minutes"] and 0 <= secs <= self.config["timer_max_seconds"]:
                        return f"{mins:02d}:{secs:02d}"
        except Exception as e:
            logger.error(f"Error in timer PaddleOCR: {e}")
        
        return None

    def _read_score_at(self, sec, container, video_stream):
        """Single-frame OCR seek at `sec`. Returns (score, timer_str) tuple."""
        time_base = float(video_stream.time_base)
        t = max(0.0, min(float(sec), self.duration - 1))
        pts = int(t / time_base)
        try:
            container.seek(pts, stream=video_stream)
            for frame in container.decode(video=0):
                img = frame.to_ndarray(format="bgr24")

                # Check if scoreboard is present using fast template comparison
                if not self._is_scoreboard_present(img):
                    return None, None

                timer_str = self._read_timer_from_frame(img)

                if self.is_split_score:
                    hx, hy, hw, hh = self.home_score_bbox_abs
                    ax, ay, aw, ah = self.away_score_bbox_abs
                    score = self._perform_ocr(
                        img[hy : hy + hh, hx : hx + hw],
                        img[ay : ay + ah, ax : ax + aw],
                        sec=sec,
                    )
                else:
                    sx, sy, sw, sh = self.score_bbox_abs
                    score = self._perform_ocr(img[sy : sy + sh, sx : sx + sw], sec=sec)

                return score, timer_str
        except Exception:
            pass
        return None, None

    def track(self):
        """Detects goal events using a simple sweep-scan at processing_fps rate.

        Uses template matching to skip processing frames where the scoreboard HUD is occluded/absent.
        Uses a 1-point difference guard to only accept goals where the score increments by exactly 1.
        """
        logger.info(
            f"Starting goal detection via sweep-scan at {self.config['processing_fps']} fps..."
        )

        container = av.open(self.video_path)
        try:
            video_stream = container.streams.video[0]
            video_stream.thread_type = "AUTO"

            goal_events = []

            # --- Step 1: Establish initial score ---
            start_t = float(self.config.get("start_time", 0.0))
            end_t = float(self.config.get("end_time") or self.duration)

            score_start_res = self._read_score_at(start_t, container, video_stream)
            score_start = score_start_res[0] if score_start_res else None

            if score_start is None:
                logger.info(
                    f"Could not read score at t={start_t} — seeking forward for first valid stable score..."
                )
                for t_check in range(int(start_t) + 1, min(int(start_t) + 100, int(end_t))):
                    score_start_res = self._read_score_at(t_check, container, video_stream)
                    score_start = score_start_res[0] if score_start_res else None
                    if score_start is not None:
                        break
                if score_start is None:
                    logger.warning(
                        "Could not find any valid score at start — defaulting to (0,0)."
                    )
                    score_start = (0, 0)

            logger.info(f"Initial accepted score: {score_start[0]}-{score_start[1]}")
            self.last_accepted_score = score_start

            # --- Step 2: Sweep through the video ---
            step = 1.0 / self.config["processing_fps"]
            timestamps = np.arange(start_t, end_t, step)

            for t in tqdm(timestamps, desc="Sweeping video"):
                t = float(t)
                score, timer_str = self._read_score_at(t, container, video_stream)
                if score is None:
                    continue

                lh, la = self.last_accepted_score
                h, a = score

                d_home = h - lh
                d_away = a - la

                # 1. Reversion detection (score decreased)
                if d_home < 0 or d_away < 0:
                    # Guard: only accept decrease if it is exactly -1 for one side and 0 for the other.
                    if not ((d_home == -1 and d_away == 0) or (d_home == 0 and d_away == -1)):
                        logger.warning(
                            f"Ignored anomalous score decrease at {t:.1f}s: {lh}-{la} -> {h}-{a}"
                        )
                        continue

                    reverted = False
                    reverted_side = "home" if d_home < 0 else "away"
                    var_window = self.config["var_window"]

                    # Check recent active goals to see if they match the reversion
                    for idx in reversed(range(len(goal_events))):
                        g = goal_events[idx]
                        if (
                            g["scorer_side"] == reverted_side
                            and (t - g["timestamp_sec"]) <= var_window
                        ):
                            # Revert this goal
                            goal_events.pop(idx)
                            logger.warning(
                                f"VAR DISALLOWED GOAL DETECTED: Goal at {g['timestamp_formatted']} ({g['timestamp_sec']}s) "
                                f"was DISALLOWED at {int(t // 60):02d}:{int(t % 60):02d} ({t:.1f}s). Reverting score to {h}-{a}."
                            )
                            self.last_accepted_score = (h, a)
                            reverted = True
                            break

                    if not reverted:
                        logger.warning(
                            f"Unmatched score decrease at {t:.1f}s: {lh}-{la} -> {h}-{a}"
                        )
                        self.last_accepted_score = (h, a)

                # 2. Guard: Only accept if the difference is exactly +1 point for either home or away,
                # and the other side has not changed.
                elif (d_home == 1 and d_away == 0) or (d_home == 0 and d_away == 1):
                    scorer = "home" if d_home == 1 else "away"
                    ts_fmt = f"{int(t // 60):02d}:{int(t % 60):02d}"

                    event = {
                        "timestamp_sec": round(t, 2),
                        "timestamp_formatted": timer_str if timer_str else ts_fmt,
                        "before_score": f"{lh}-{la}",
                        "after_score": f"{h}-{a}",
                        "scorer_side": scorer,
                    }
                    goal_events.append(event)
                    logger.info(
                        f"GOAL DETECTED: {lh}-{la} -> {h}-{a} (Scorer: {scorer}) at {timer_str if timer_str else ts_fmt} ({t:.1f}s)"
                    )

                    # Update last accepted score
                    self.last_accepted_score = (h, a)

            # --- Step 3: Seek for before/after frames and save collages ---
            # ======DEBUG======
            if self.debug_dir:
                for event in goal_events:
                    t = event["timestamp_sec"]
                    before_img = after_img = None
                    for seek_t, label in (
                        (max(0.0, t - 2), "before"),
                        (min(self.duration - 1, t + 2), "after"),
                    ):
                        pts = int(seek_t / float(video_stream.time_base))
                        try:
                            container.seek(pts, stream=video_stream)
                            for frame in container.decode(video=0):
                                if label == "before":
                                    before_img = frame.to_ndarray(format="bgr24")
                                else:
                                    after_img = frame.to_ndarray(format="bgr24")
                                break
                        except Exception:
                            pass

                    if before_img is not None and after_img is not None:
                        collage_filename = (
                            f"goal_{event['before_score']}_to_"
                            f"{event['after_score']}_{int(t)}.png"
                        )
                        save_goal_collage(
                            before_img,
                            after_img,
                            event["before_score"],
                            event["after_score"],
                            t,
                            os.path.join(self.output_dir, collage_filename),
                        )
            # ======DEBUG======

            container.close()
            logger.info(f"Tracking complete. Total goals detected: {len(goal_events)}")

            # --- Step 4: Run Replay Detection for each detected goal ---
            if self.config.get("debug_tracking", False):
                logger.info("Debug tracking flag enabled. Skipping replay detection phase.")
            elif goal_events and not self.config.get("no_replay", False):
                import subprocess

                logger.info("Initializing Replay Detection for detected goal events...")
                try:
                    # Build configuration for replay detector
                    replay_config = {
                        "strong_th": self.config.get("replay_strong_th", 0.15),
                        "green_th": self.config.get("replay_green_th", 0.70),
                        "replay_search_buffer": self.config.get(
                            "replay_search_buffer", 60.0
                        ),
                        "replay_search_end_buffer": self.config.get(
                            "replay_search_end_buffer", 60.0
                        ),
                        "min_absence_duration": self.config.get(
                            "min_absence_duration", 0.5
                        ),
                        "min_absence_priority_duration": self.config.get(
                            "min_absence_priority_duration", 10.0
                        ),
                    }
                    replay_det = ReplayDetector(self.video_path, replay_config)

                    export_clips = self.config.get("replay_export", True)

                    for idx, event in enumerate(goal_events):
                        anchor = event["timestamp_sec"]
                        logger.info(
                            f"Scanning for replay of Goal #{idx + 1} with anchor {anchor}s..."
                        )

                        # Pre-compute scoreboard absence periods in the replay window
                        scan_start = max(
                            0.0, anchor - replay_config["replay_search_buffer"]
                        )
                        scan_end = anchor + replay_config["replay_search_end_buffer"]
                        absence_periods = self.scan_scoreboard_presence(
                            scan_start, scan_end
                        )

                        res = replay_det.detect_replay_bounds(
                            anchor, absence_periods=absence_periods, show_progress=False
                        )
                        if res:
                            event["replay"] = {
                                "true_start_frame": res["true_start_frame"],
                                "true_end_frame": res["true_end_frame"],
                                "true_start_sec": round(res["true_start_sec"], 2),
                                "true_end_sec": round(res["true_end_sec"], 2),
                                "duration_sec": round(res["duration_sec"], 2),
                                "entrance_wipe_detected": res["entrance_wipe"] is not None,
                                "exit_wipe_detected": res["exit_wipe"] is not None,
                                "refinement_source": res.get(
                                    "refinement_source", "unknown"
                                ),
                            }
                            logger.info(
                                f"Replay detected: {res['true_start_sec']:.2f}s -> {res['true_end_sec']:.2f}s (dur: {res['duration_sec']:.2f}s)"
                            )

                            if export_clips and self.debug_dir is not None:
                                clip_filename = f"replay_goal_{event['before_score']}_to_{event['after_score']}_{int(anchor)}.mp4"
                                clip_path = os.path.join(self.output_dir, clip_filename)
                                logger.info(f"Exporting replay clip to {clip_path}...")

                                cmd = [
                                    "ffmpeg",
                                    "-y",
                                    "-ss",
                                    f"{res['true_start_sec']:.3f}",
                                    "-i",
                                    self.video_path,
                                    "-t",
                                    f"{res['duration_sec']:.3f}",
                                    "-c",
                                    "copy",
                                    clip_path,
                                ]
                                try:
                                    subprocess.run(
                                        cmd,
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL,
                                        check=True,
                                    )
                                    event["replay"]["clip_path"] = clip_path
                                    logger.info(
                                        f"Successfully exported replay clip: {clip_path}"
                                    )
                                except Exception as e_ffmpeg:
                                    logger.warning(
                                        f"Failed to export replay clip using FFmpeg: {e_ffmpeg}"
                                    )
                        else:
                            logger.info(
                                f"No replay candidate found for Goal #{idx + 1} after anchor {anchor}s."
                            )
                except Exception as e_replay:
                    logger.error(f"Error running replay detection pipeline: {e_replay}")
        finally:
            container.close()
        logger.info(
            f"Total OCR runs: {self.ocr_run_count}, skips: {self.ocr_skip_count}"
        )

        # ======DEBUG======
        if self.debug_dir:
            # Save goal events to JSON
            with open(os.path.join(self.output_dir, "goals.json"), "w") as f:
                json.dump(goal_events, f, indent=4)
        # ======DEBUG======

        return goal_events
