import os
import json
import logging
import cv2
import numpy as np

from src.config import REPLAY_DETECTOR_DEFAULT_CONFIG

logger = logging.getLogger('goal_detector')

def normalize_array(arr):
    arr = np.array(arr, dtype=float)
    amin, amax = np.min(arr), np.max(arr)
    denom = amax - amin
    if denom < 1e-8:
        return np.zeros_like(arr)
    return (arr - amin) / denom

class ReplayDetector:
    def __init__(self, video_path, config=None):
        self.video_path = video_path
        
        # Default configuration parameters
        self.config = REPLAY_DETECTOR_DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
            
        self.fps = None
        self.total_frames = None
        self._load_video_metadata()

    def _load_video_metadata(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        try:
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        logger.info(f"ReplayDetector: video loaded ({self.total_frames} frames, {self.fps:.2f} fps)")

    def get_match_green_stats(self, cache_path=None):
        if hasattr(self, '_mean_g'):
            return self._mean_g, self._median_g
            
        if cache_path is None:
            # Let's put it in the same place as the project's output/ folder
            # Find the root of /home/laffey/Projects/work/ocr
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            cache_path = os.path.join(root, "output", "green_ratio_cache.json")
            
        debug_mode = os.environ.get("DEBUG", "").lower() in ("1", "true", "yes", "y")
        cache = {}
        if debug_mode and os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cache = json.load(f)
            except Exception:
                pass
                
        abs_video_path = os.path.abspath(self.video_path)
        if abs_video_path in cache:
            self._mean_g = cache[abs_video_path]['mean']
            self._median_g = cache[abs_video_path]['median']
            return self._mean_g, self._median_g
            
        logger.info(f"Computing green ratio stats across the match for {self.video_path}...")
        cap = cv2.VideoCapture(self.video_path)
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                self._mean_g, self._median_g = 0.5, 0.5
                return 0.5, 0.5
                
            n_samples = 200
            indices = np.linspace(0, total_frames - 1, n_samples, dtype=int)
            green_ratios = []
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret:
                    continue
                h, w = frame.shape[:2]
                scale = 320.0 / w
                small_frame = cv2.resize(frame, (320, int(round(h * scale))))
                hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
                g, _, _, _ = self.get_hsv_ratios(hsv)
                green_ratios.append(g)
        finally:
            cap.release()
        
        if not green_ratios:
            self._mean_g, self._median_g = 0.5, 0.5
            return 0.5, 0.5
            
        self._mean_g = float(np.mean(green_ratios))
        self._median_g = float(np.median(green_ratios))
        
        if debug_mode:
            cache[abs_video_path] = {'mean': self._mean_g, 'median': self._median_g}
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            try:
                with open(cache_path, 'w') as f:
                    json.dump(cache, f, indent=2)
            except Exception:
                pass
            
        return self._mean_g, self._median_g

    def get_hsv_ratios(self, hsv_frame):
        """Computes ratios of green, blue, red, and white pixels in an HSV frame."""
        lower_green = np.array(self.config['green_hsv_lower'])
        upper_green = np.array(self.config['green_hsv_upper'])
        green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
        green_ratio = np.sum(green_mask > 0) / green_mask.size
        
        lower_blue = np.array(self.config['blue_hsv_lower'])
        upper_blue = np.array(self.config['blue_hsv_upper'])
        blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
        blue_ratio = np.sum(blue_mask > 0) / blue_mask.size
        
        lower_red1 = np.array(self.config['red_hsv_lower1'])
        upper_red1 = np.array(self.config['red_hsv_upper1'])
        lower_red2 = np.array(self.config['red_hsv_lower2'])
        upper_red2 = np.array(self.config['red_hsv_upper2'])
        red_mask = cv2.inRange(hsv_frame, lower_red1, upper_red1) | cv2.inRange(hsv_frame, lower_red2, upper_red2)
        red_ratio = np.sum(red_mask > 0) / red_mask.size
        
        lower_white = np.array(self.config['white_hsv_lower'])
        upper_white = np.array(self.config['white_hsv_upper'])
        white_mask = cv2.inRange(hsv_frame, lower_white, upper_white)
        white_ratio = np.sum(white_mask > 0) / white_mask.size
        
        return green_ratio, blue_ratio, red_ratio, white_ratio



    def _compute_block_max_nongreen_delta(self, hsv_curr, hsv_prev, n_rows=8, n_cols=8):
        h, w = hsv_curr.shape[:2]
        bh = h // n_rows
        bw = w // n_cols

        lower_green = np.array(self.config['green_hsv_lower'])
        upper_green = np.array(self.config['green_hsv_upper'])

        max_delta = 0.0
        for r in range(n_rows):
            for c in range(n_cols):
                y0, y1 = r * bh, (r + 1) * bh
                x0, x1 = c * bw, (c + 1) * bw

                block_curr = hsv_curr[y0:y1, x0:x1]
                block_prev = hsv_prev[y0:y1, x0:x1]

                size = block_curr.size // 3
                if size == 0:
                    continue

                green_curr = np.sum(cv2.inRange(block_curr, lower_green, upper_green) > 0) / size
                green_prev = np.sum(cv2.inRange(block_prev, lower_green, upper_green) > 0) / size

                delta = abs((1.0 - green_curr) - (1.0 - green_prev))
                if delta > max_delta:
                    max_delta = delta

        return max_delta


    def _refine_with_scoreboard(self, anchor_sec, absence_periods,
                                 fallback_start_frame, fallback_end_frame,
                                 min_priority_dur=None):
        """Uses scoreboard absence periods to refine replay boundaries.

        Scoreboard absence is treated as the primary (most accurate) signal for
        replay detection. Falls back to None if no valid absence period overlaps
        the candidate scene block.

        Returns:
            (start_frame, end_frame, total_absence_duration) or None if no valid refinement found.
        """
        if not absence_periods:
            return None

        min_dur = self.config['min_absence_duration']
        block_start_sec = fallback_start_frame / self.fps
        block_end_sec = fallback_end_frame / self.fps

        # Find absence periods that overlap the scene block
        overlapping = []
        for abs_start, abs_end in absence_periods:
            abs_dur = abs_end - abs_start
            # Skip candidate periods that are shorter than the priority duration if specified (e.g. flickers/animations)
            if min_priority_dur is not None and abs_dur < min_priority_dur:
                logger.info(f"ReplayDetector: skipping short scoreboard absence candidate [{abs_start:.2f}s - {abs_end:.2f}s] "
                            f"(duration {abs_dur:.2f}s < min_priority_dur {min_priority_dur:.2f}s)")
                continue

            # Check overlap
            overlap_start = max(abs_start, block_start_sec)
            overlap_end = min(abs_end, block_end_sec)
            if overlap_start < overlap_end:
                overlapping.append((abs_start, abs_end, overlap_start, overlap_end))

        if not overlapping:
            logger.info("ReplayDetector: no scoreboard absence periods overlap the candidate block")
            return None

        # Pick the absence period whose midpoint is closest to the anchor
        best = None
        best_dist = float('inf')
        for abs_start, abs_end, ov_start, ov_end in overlapping:
            mid = (abs_start + abs_end) / 2.0
            dist = abs(mid - anchor_sec)
            dur = ov_end - ov_start
            if dist < best_dist and dur >= min_dur:
                best_dist = dist
                best = (abs_start, abs_end, ov_start, ov_end)

        if best is None:
            logger.info(f"ReplayDetector: {len(overlapping)} absence(s) overlap but all < {min_dur:.1f}s")
            return None

        abs_start, abs_end, ov_start, ov_end = best
        start_frame = int(round(ov_start * self.fps))
        end_frame = int(round(ov_end * self.fps))

        logger.info(f"ReplayDetector: scoreboard absence refinement = "
                     f"[{ov_start:.2f}s - {ov_end:.2f}s] "
                     f"(frames {start_frame}-{end_frame}, "
                     f"dist_to_anchor={best_dist:.2f}s)")

        return (start_frame, end_frame, abs_end - abs_start)

    def detect_replay_bounds(self, anchor_sec, absence_periods=None, show_progress=False):
        """Executes the replay detection pipeline starting from an anchor timestamp."""
        logger.info(f"ReplayDetector: detecting replay bounds (anchor={anchor_sec:.2f}s) without PySceneDetect...")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error("ReplayDetector: could not open video for wipe scanning")
            return None
        try:
            return self._detect_replay_bounds_impl(cap, anchor_sec, absence_periods, show_progress)
        finally:
            cap.release()

    def _detect_replay_bounds_impl(self, cap, anchor_sec, absence_periods=None, show_progress=False):
        # We search from anchor_sec to anchor_sec + search_end_buffer + 5.0s (default 65s after anchor)
        buf_after = self.config.get('replay_search_end_buffer', 60.0)
        scan_start_sec = max(0.0, anchor_sec)
        scan_end_sec = min(self.total_frames / self.fps, anchor_sec + buf_after + 5.0)
        
        start_frame = int(round(scan_start_sec * self.fps))
        end_frame = int(round(scan_end_sec * self.fps))
        
        logger.info(f"ReplayDetector: scanning window [{scan_start_sec:.1f}s - {scan_end_sec:.1f}s] "
                     f"(frames {start_frame}-{end_frame}) for transitions...")
                     
        # 2. Coarse frame scan (step = 3) — collect green ratios + block_max_ng across the full search window
        coarse_cache = {}
        coarse_prev_hsv = None
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for f_idx in range(start_frame, end_frame):
            if (f_idx - start_frame) % 3 == 0:
                ret, frame = cap.read()
                if not ret:
                    break
                h, w = frame.shape[:2]
                scale = 320.0 / w
                small_frame = cv2.resize(frame, (320, int(round(h * scale))))
                hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
                g, b, r, w_ratio = self.get_hsv_ratios(hsv)
                block_max_ng = self._compute_block_max_nongreen_delta(hsv, coarse_prev_hsv) if coarse_prev_hsv is not None else 0.0
                coarse_cache[f_idx] = (g, b, r, w_ratio, block_max_ng)
                coarse_prev_hsv = hsv
            else:
                cap.grab()

        # 3a. Find entrance candidates using the product signal (block_max_ng × binary_drop).
        #     Search strictly FORWARD from anchor_sec.
        #     A peak > 0.5 is a potential; it becomes a full candidate if no other
        #     potential within the next 15 s is >= 85 % of its magnitude.
        anchor_frame = int(round(anchor_sec * self.fps))
        forward_sorted = sorted(f for f in coarse_cache if f >= anchor_frame)

        entrance_candidates = []  # list of (peak_frame, product_score)
        potentials = []  # list of (i, frame_idx, score)

        if forward_sorted:
            mean_g, _ = self.get_match_green_stats()
            mean_ng_base = 1.0 - mean_g
            stay_sec     = self.config.get('drop_stay_sec', 1.0)
            tolerance    = self.config.get('drop_tolerance', 0.1)
            # coarse stride is every 3 frames
            n_stay_coarse = max(1, int(round(stay_sec * self.fps / 3)))
            threshold_ng  = mean_ng_base + tolerance
            pulse_coarse  = max(1, int(round(1.0 * self.fps / 3)))
            half_pulse    = pulse_coarse // 2

            ng_fwd    = np.array([1.0 - coarse_cache[f][0] for f in forward_sorted])
            block_fwd = np.array([coarse_cache[f][4]       for f in forward_sorted])
            n_fwd     = len(ng_fwd)

            # Binary drop signal (strict): pulse fires at drop-onset frame i
            binary_fwd = np.zeros(n_fwd)
            i = 1
            while i < n_fwd:
                if ng_fwd[i-1] > mean_ng_base and ng_fwd[i] <= threshold_ng and ng_fwd[i] < ng_fwd[i-1]:
                    j = i
                    while j < n_fwd and ng_fwd[j] <= threshold_ng:
                        j += 1
                    if (j - i) >= n_stay_coarse:
                        s_idx = max(0, i - half_pulse)
                        e_idx = min(n_fwd, s_idx + pulse_coarse)
                        binary_fwd[s_idx:e_idx] = 1.0
                        k = i + 1
                        while k < n_fwd and ng_fwd[k] <= mean_ng_base:
                            k += 1
                        i = max(i + n_stay_coarse, k)
                        continue
                i += 1

            product_fwd = block_fwd * binary_fwd

            # Collect peaks above threshold
            product_peak_th  = 0.37
            peak_fwd_coarse  = int(round(15.0 * self.fps / 3))  # 15 s in coarse-frame units
            potentials = [
                (i, forward_sorted[i], float(product_fwd[i]))
                for i in range(n_fwd)
                if product_fwd[i] > product_peak_th
            ]
            logger.info(
                f"ReplayDetector: product signal found {len(potentials)} potential entrance peak(s) "
                f"> {product_peak_th} (searching forward from {anchor_sec:.1f}s)"
            )

            # Upgrade to full candidate: valid if no other potential in [i+1, i+15s] has >= 85 % of its score
            for (i, frame_idx, score) in potentials:
                dominated = any(
                    (i < j <= i + peak_fwd_coarse) and score_j >= 0.90 * score
                    for (j, _fj, score_j) in potentials
                )
                if not dominated:
                    entrance_candidates.append((frame_idx, score))
                    logger.info(
                        f"  Entrance candidate: frame={frame_idx} "
                        f"({frame_idx / self.fps:.2f}s), product_score={score:.3f}"
                    )

            logger.info(
                f"ReplayDetector: {len(entrance_candidates)} entrance candidate(s) after 15s/85% filter"
            )

            # Safety expansion: ensure we have scanned at least 60 seconds after every upgraded candidate
            max_required_sec = 0.0
            for ec_frame, _ in entrance_candidates:
                required_sec = (ec_frame / self.fps) + 75.0
                if required_sec > max_required_sec:
                    max_required_sec = required_sec
            
            max_video_sec = self.total_frames / self.fps
            max_required_sec = min(max_video_sec, max_required_sec)
            
            if max_required_sec > scan_end_sec:
                logger.info(
                    f"ReplayDetector: safety expansion — expanding scan window end from {scan_end_sec:.1f}s to {max_required_sec:.1f}s "
                    f"to ensure 60s coverage after upgraded entrance candidate(s)"
                )
                new_end_frame = int(round(max_required_sec * self.fps))
                
                # Resume coarse scanning from the current end_frame (we need to restore coarse_prev_hsv)
                last_scanned = max(coarse_cache.keys()) if coarse_cache else start_frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, last_scanned)
                if last_scanned in coarse_cache:
                    ret, frame = cap.read()
                    if ret:
                        h, w = frame.shape[:2]
                        scale = 320.0 / w
                        small_frame = cv2.resize(frame, (320, int(round(h * scale))))
                        coarse_prev_hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
                
                # Scan from last_scanned + 1 to new_end_frame
                for f_idx in range(last_scanned + 1, new_end_frame):
                    if (f_idx - start_frame) % 3 == 0:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        h, w = frame.shape[:2]
                        scale = 320.0 / w
                        small_frame = cv2.resize(frame, (320, int(round(h * scale))))
                        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
                        g, b, r, w_ratio = self.get_hsv_ratios(hsv)
                        block_max_ng = self._compute_block_max_nongreen_delta(hsv, coarse_prev_hsv) if coarse_prev_hsv is not None else 0.0
                        coarse_cache[f_idx] = (g, b, r, w_ratio, block_max_ng)
                        coarse_prev_hsv = hsv
                    else:
                        cap.grab()
                
                scan_end_sec = max_required_sec
                end_frame = new_end_frame

        # 3b. Identify coarse candidate frames using a rolling window peak-to-bottom strategy.
        mean_g, _ = self.get_match_green_stats()
        mean_ng_base = 1.0 - mean_g
        relaxed_gate = mean_ng_base + 0.3

        coarse_sorted = sorted(coarse_cache.keys())
        coarse_ng = np.array([1.0 - coarse_cache[f][0] for f in coarse_sorted])
        n_coarse = len(coarse_ng)

        # 5 seconds rolling window in coarse frames (every 3 frames)
        win_size_coarse = max(3, int(round(5.0 * self.fps / 3)))
        stay_low_coarse = max(3, int(round(5.0 * self.fps / 3)))

        exit_candidates = []
        for i in range(n_coarse - win_size_coarse):
            window_ng = coarse_ng[i : i + win_size_coarse]
            idx_max = int(np.argmax(window_ng))
            idx_min = int(np.argmin(window_ng))
            
            if idx_max < idx_min:
                peak_val = window_ng[idx_max]
                bottom_val = window_ng[idx_min]
                
                if peak_val > mean_ng_base and bottom_val < relaxed_gate:
                    # Check stay low for 5s starting from the bottom point index
                    start_k = i + idx_min
                    end_k = min(n_coarse, start_k + stay_low_coarse)
                    stay_low = True
                    for k in range(start_k, end_k):
                        if coarse_ng[k] > relaxed_gate:
                            stay_low = False
                            break
                            
                    if stay_low:
                        peak_frame = coarse_sorted[i + idx_max]
                        drop_amount = peak_val - bottom_val
                        exit_candidates.append({
                            'peak_frame': peak_frame,
                            'drop_amount': drop_amount
                        })

        # Non-Maximum Suppression (NMS) within 5.0s window
        exit_candidates = sorted(exit_candidates, key=lambda x: x['peak_frame'])
        filtered_exit_candidates = []
        for cand in exit_candidates:
            if not filtered_exit_candidates:
                filtered_exit_candidates.append(cand)
            else:
                prev = filtered_exit_candidates[-1]
                time_diff = (cand['peak_frame'] - prev['peak_frame']) / self.fps
                if time_diff < 5.0:
                    if cand['drop_amount'] > prev['drop_amount']:
                        filtered_exit_candidates[-1] = cand
                else:
                    filtered_exit_candidates.append(cand)
        candidate_frames = [cand['peak_frame'] for cand in filtered_exit_candidates]

        # Include starting/entrance candidates as potential exit candidates too
        exit_candidate_frames = list(candidate_frames)
        for _, pot_frame, _ in potentials:
            if pot_frame not in exit_candidate_frames:
                exit_candidate_frames.append(pot_frame)

        logger.info(
            f"ReplayDetector: rolling-window detector found {len(candidate_frames)} candidate frame(s) "
            f"matching peak-to-bottom strategy. Total exit candidates (including entrance potentials): {len(exit_candidate_frames)}"
        )

        # 4. Fine-grained scans (step=1) around each candidate
        #    We scan +/- 1.0 seconds around each coarse candidate
        win_frames = int(round(1.0 * self.fps))
        fine_ent_intervals = [
            (max(start_frame, ec_frame - win_frames), min(end_frame, ec_frame + win_frames))
            for ec_frame, _ in entrance_candidates
        ]
        fine_ex_intervals = [
            (max(start_frame, cf - win_frames), min(end_frame, cf + win_frames))
            for cf in exit_candidate_frames
        ]

        all_fine_intervals = fine_ent_intervals + fine_ex_intervals

        # Merge overlapping fine-scan intervals
        merged_fine_intervals = []
        if all_fine_intervals:
            sorted_fine = sorted(all_fine_intervals, key=lambda x: x[0])
            merged_fine_intervals = [sorted_fine[0]]
            for current in sorted_fine[1:]:
                prev = merged_fine_intervals[-1]
                if current[0] <= prev[1]:
                    merged_fine_intervals[-1] = (prev[0], max(prev[1], current[1]))
                else:
                    merged_fine_intervals.append(current)

        # Populate frames_cache with step-1 scans on merged intervals (with block-max delta)
        frames_cache = {}
        for f_start, f_end in merged_fine_intervals:
            prev_hsv = None
            if f_start > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_start - 1)
                ret, frame = cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    scale = 320.0 / w
                    small_frame = cv2.resize(frame, (320, int(round(h * scale))))
                    prev_hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)

            cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
            for f_idx in range(f_start, f_end):
                ret, frame = cap.read()
                if not ret:
                    break
                h, w = frame.shape[:2]
                scale = 320.0 / w
                small_frame = cv2.resize(frame, (320, int(round(h * scale))))
                hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
                g, b, r, w_ratio = self.get_hsv_ratios(hsv)
                
                block_max_ng = self._compute_block_max_nongreen_delta(hsv, prev_hsv) if prev_hsv is not None else 0.0
                
                frames_cache[f_idx] = (g, b, r, w_ratio, block_max_ng)
                prev_hsv = hsv

        # Refine entrance candidates: find exact frame with max block_max_ng in +/- 1.0 second range
        refined_entrance_wipes = []
        for ec_frame, ec_score in entrance_candidates:
            f_start = max(start_frame, ec_frame - win_frames)
            f_end = min(end_frame, ec_frame + win_frames)
            sorted_frames = sorted(f for f in frames_cache if f_start <= f < f_end)
            if sorted_frames:
                block_maxes = [frames_cache[f][4] for f in sorted_frames]
                best_idx = int(np.argmax(block_maxes))
                best_val = float(block_maxes[best_idx])
                refined_frame = sorted_frames[best_idx]
                refined_entrance_wipes.append({
                    'peak_frame':     refined_frame,
                    'peak_ratio':     best_val if best_val > 0.0 else ec_score,
                    'peak_color':     'product_signal',
                    'step_score':     0.0,
                    'combined_score': best_val if best_val > 0.0 else ec_score,
                    'start_frame':    refined_frame,
                    'end_frame':      refined_frame,
                    'length':         1,
                })
            else:
                refined_entrance_wipes.append({
                    'peak_frame':     ec_frame,
                    'peak_ratio':     ec_score,
                    'peak_color':     'product_signal',
                    'step_score':     0.0,
                    'combined_score': ec_score,
                    'start_frame':    ec_frame,
                    'end_frame':      ec_frame,
                    'length':         1,
                })

        # Refine exit candidates: find exact frame with max block_max_ng in +/- 1.0 second range
        refined_exit_wipes = []
        for cf in exit_candidate_frames:
            f_start = max(start_frame, cf - win_frames)
            f_end = min(end_frame, cf + win_frames)
            sorted_frames = sorted(f for f in frames_cache if f_start <= f < f_end)
            if sorted_frames:
                block_maxes = [frames_cache[f][4] for f in sorted_frames]
                best_idx = int(np.argmax(block_maxes))
                best_val = float(block_maxes[best_idx])
                refined_frame = sorted_frames[best_idx]
                refined_exit_wipes.append({
                    'peak_frame':     refined_frame,
                    'peak_ratio':     best_val,
                    'peak_color':     'lenient_exit_product',
                    'step_score':     0.0,
                    'combined_score': best_val,
                    'start_frame':    refined_frame,
                    'end_frame':      refined_frame,
                    'length':         1,
                })

        # Deduplicate refined exit wipes by peak_frame to handle overlapping candidates
        deduped_exit_wipes = {}
        for xw in refined_exit_wipes:
            pf = xw['peak_frame']
            if pf not in deduped_exit_wipes or xw['peak_ratio'] > deduped_exit_wipes[pf]['peak_ratio']:
                deduped_exit_wipes[pf] = xw
        refined_exit_wipes = list(deduped_exit_wipes.values())

        # 5. Pair each entrance candidate with the best exit.
        entrance_wipe = None
        exit_wipe = None
        best_score = -float('inf')

        def _absence_bonus(ec_s, exit_s, dur_s, abs_periods):
            if dur_s < self.config['optimal_replay_duration_min']:
                return 0.0
            if not abs_periods:
                return 0.0
            total_overlap = 0.0
            for abs_start, abs_end in abs_periods:
                if (abs_end - abs_start) < self.config['absence_bonus_duration_th']:
                    continue
                ov = min(exit_s, abs_end) - max(ec_s, abs_start)
                if ov > 0:
                    total_overlap += ov
            if total_overlap >= self.config['absence_bonus_min_overlap'] or total_overlap >= self.config['absence_bonus_overlap_ratio'] * dur_s:
                return self.config['absence_bonus_max_val']
            return self.config['absence_bonus_mid_val'] if total_overlap > 0 else 0.0

        def _duration_penalty(dur_s):
            if self.config['optimal_replay_duration_min'] <= dur_s <= self.config['optimal_replay_duration_max']:
                return 0.0
            elif dur_s < self.config['duration_penalty_short_tol']:
                return self.config['duration_penalty_short_multiplier'] * (self.config['duration_penalty_short_ref'] - dur_s)
            else:
                return self.config['duration_penalty_long_multiplier'] * (dur_s - self.config['duration_penalty_long_ref'])

        if refined_entrance_wipes:
            for ew in refined_entrance_wipes:
                ec_frame = ew['peak_frame']
                ec_sec = ec_frame / self.fps
                ec_product_score = ew['peak_ratio']

                for xw in refined_exit_wipes:
                    ex_frame = xw['peak_frame']
                    w_exit_sec = ex_frame / self.fps
                    dur_sec = w_exit_sec - ec_sec
                    if not (self.config['min_replay_duration'] <= dur_sec <= self.config['max_replay_duration']):
                        continue

                    pair_score = (ec_product_score
                                  + xw['peak_ratio']
                                  + _absence_bonus(ec_sec, w_exit_sec, dur_sec, absence_periods)
                                  - _duration_penalty(dur_sec)
                                  - self.config['anchor_delay_penalty_weight'] * max(0.0, ec_sec - anchor_sec))

                    if pair_score > best_score:
                        best_score  = pair_score
                        entrance_wipe = ew
                        exit_wipe   = xw

        if entrance_wipe and exit_wipe:
            logger.info(
                f"ReplayDetector: selected matching wipe pair: "
                f"entrance={entrance_wipe['peak_frame']} "
                f"(product_score={entrance_wipe['peak_ratio']:.3f}), "
                f"exit={exit_wipe['peak_frame']} "
                f"(ratio={exit_wipe['peak_ratio']:.3f}) | "
                f"duration={exit_wipe['peak_frame']/self.fps - entrance_wipe['peak_frame']/self.fps:.2f}s, "
                f"score={best_score:.3f}"
            )
                        
        # 6. Compute true start and end
        # Fallback to default replay block [anchor_sec, anchor_sec + fallback_replay_duration] if no wipes are detected
        fallback_start_frame = int(round(anchor_sec * self.fps))
        fallback_end_frame = int(round((anchor_sec + self.config['fallback_replay_duration']) * self.fps))
        
        if entrance_wipe or exit_wipe:
            # Cut boundaries: start is after entrance wipe ends, end is before exit wipe starts
            baseline_start = entrance_wipe['end_frame'] if entrance_wipe else fallback_start_frame
            baseline_end = exit_wipe['start_frame'] if exit_wipe else fallback_end_frame
            baseline_source = "wipe_detection"
        else:
            baseline_start = fallback_start_frame
            baseline_end = fallback_end_frame
            baseline_source = "default_fallback"
            
        true_start_frame = baseline_start
        true_end_frame = baseline_end
        refinement_source = baseline_source

        # Refine with scoreboard absence if it overlaps the baseline bounds significantly
        if absence_periods:
            sb_result = self._refine_with_scoreboard(
                anchor_sec, absence_periods,
                start_frame, end_frame,
                min_priority_dur=(self.config.get('min_absence_priority_duration', 10.0) if baseline_source == "wipe_detection" else None)
            )
            if sb_result is not None:
                sb_start, sb_end, sb_abs_dur = sb_result
                
                # Check overlap between scoreboard absence and baseline bounds
                overlap_start = max(sb_start, baseline_start)
                overlap_end = min(sb_end, baseline_end)
                overlap_dur = max(0.0, (overlap_end - overlap_start + 1) / self.fps) if overlap_end >= overlap_start else 0.0
                
                baseline_dur = (baseline_end - baseline_start + 1) / self.fps
                is_fallback = (baseline_source == "default_fallback")
                significant_overlap = (overlap_dur >= 5.0) or (overlap_dur >= 0.5 * baseline_dur)
                
                min_priority_dur = self.config.get('min_absence_priority_duration', 10.0)
                
                allow_refinement = False
                if is_fallback:
                    allow_refinement = significant_overlap
                else:
                    if sb_abs_dur >= min_priority_dur:
                        allow_refinement = significant_overlap
                    else:
                        logger.info(
                            f"ReplayDetector: ignoring scoreboard absence refinement [{sb_start/self.fps:.2f}s - {sb_end/self.fps:.2f}s] "
                            f"because its total absence duration ({sb_abs_dur:.2f}s) is less than "
                            f"min_absence_priority_duration ({min_priority_dur:.2f}s)"
                        )
                
                if allow_refinement:
                    true_start_frame = sb_start
                    true_end_frame = sb_end
                    refinement_source = "scoreboard_absence"

        true_start_sec = true_start_frame / self.fps
        true_end_sec = true_end_frame / self.fps
        duration = (true_end_frame - true_start_frame + 1) / self.fps

        logger.info(f"ReplayDetector: result = [{true_start_sec:.2f}s - {true_end_sec:.2f}s] "
                     f"(duration={duration:.2f}s, source={refinement_source})")
        
        return {
            'fallback_start_frame': fallback_start_frame,
            'fallback_end_frame': fallback_end_frame,
            'entrance_wipe': entrance_wipe,
            'exit_wipe': exit_wipe,
            'true_start_frame': true_start_frame,
            'true_end_frame': true_end_frame,
            'true_start_sec': true_start_sec,
            'true_end_sec': true_end_sec,
            'duration_sec': duration,
            'refinement_source': refinement_source,
            'fps': self.fps,
            'total_frames': self.total_frames,
            'scan_start_sec': scan_start_sec,
            'scan_end_sec': scan_end_sec
        }


