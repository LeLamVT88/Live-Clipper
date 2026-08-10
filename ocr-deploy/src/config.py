import os
from dotenv import load_dotenv

# Load environment variables from a .env file if present at the root of the workspace
load_dotenv()

def get_env_bool(name, default):
    """Reads environment variable and returns a boolean value."""
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "y")

def get_env_int(name, default):
    """Reads environment variable and returns an integer value."""
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default

def get_env_float(name, default):
    """Reads environment variable and returns a float value."""
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default

def get_env_list(name, default):
    """Reads environment variable expecting a comma-separated list of integers."""
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return [int(x.strip()) for x in val.split(",")]
    except ValueError:
        return default

# ==============================================================================
# Goal Detector Default Configuration Parameters
# ==============================================================================
GOAL_DETECTOR_DEFAULT_CONFIG = {
    # Number of frames to sample evenly across the video during calibration
    'sample_n_frames': get_env_int('SAMPLE_N_FRAMES', 200),
    
    # Number of stable/clean frames to keep for the calibration process
    'clean_frame_keep': get_env_int('CLEAN_FRAME_KEEP', 120),
    
    # Width fraction of the frame to use as search ROI for scoreboard localization
    'search_roi_w_frac': get_env_float('SEARCH_ROI_W_FRAC', 0.40),
    
    # Height fraction of the frame to use as search ROI for scoreboard localization
    'search_roi_h_frac': get_env_float('SEARCH_ROI_H_FRAC', 0.30),
    
    # Block size (in pixels) for block-level temporal variance calculations
    'block_size': get_env_int('BLOCK_SIZE', 8),
    
    # Variance percentile threshold for identifying scoreboard candidates (top % stable pixels)
    'variance_percentile': get_env_int('VARIANCE_PERCENTILE', 10),
    
    # Minimum aspect ratio (width/height) for localized score/scoreboard box candidates
    'score_aspect_min': get_env_float('SCORE_ASPECT_MIN', 2.0),
    
    # Maximum aspect ratio (width/height) for localized score/scoreboard box candidates
    'score_aspect_max': get_env_float('SCORE_ASPECT_MAX', 16.0),
    
    # Minimum confidence ratio to accept localized scoreboard candidates
    'calibration_confidence_min': get_env_float('CALIBRATION_CONFIDENCE_MIN', 0.60),
    
    # Image upscaling factor used during preprocessing for PaddleOCR recognition
    'ocr_scale': get_env_int('OCR_SCALE', 3),
    
    # Allowed character list whitelist passed to OCR engines
    'ocr_whitelist': os.environ.get('OCR_WHITELIST', "0123456789-–: "),
    
    # Debounce filter buffer size to avoid score reading glitches (consecutive frames agreement)
    'debounce_k': get_env_int('DEBOUNCE_K', 3),
    
    # Number of consecutive frames the scoreboard must be absent to trigger an occlusion state
    'absent_threshold_m': get_env_int('ABSENT_THRESHOLD_M', 5),
    
    # Scanning frequency (frames per second) for the main goal tracker loop
    'processing_fps': get_env_float('PROCESSING_FPS', 0.5),
    
    # Rolling window seconds used to detect VAR goal reversion events
    'var_window': get_env_int('VAR_WINDOW', 300),
    
    # Maximum score value allowed before rejecting it as an OCR reading error
    'max_score_value': get_env_int('MAX_SCORE_VALUE', 20),
    
    # Padding fraction added around localized score bounds
    'score_box_pad_frac': get_env_float('SCORE_BOX_PAD_FRAC', 0.10),
    
    # Minimum PaddleOCR confidence score (%) required to accept a read digit
    'min_ocr_confidence': get_env_float('MIN_OCR_CONFIDENCE', 50.0),
    
    # Enable pixel-difference change detection cache to avoid calling OCR on identical frames
    'use_change_cache': get_env_bool('USE_CHANGE_CACHE', True),
    
    # Run only tracking phase (skip replay detection) and exit early
    'debug_tracking': get_env_bool('DEBUG_TRACKING', False),
    
    # Scanning frequency (frames per second) for scoreboard presence in replay window
    'scoreboard_scan_fps': get_env_float('SCOREBOARD_SCAN_FPS', 2.0),
    
    # Minimum duration (seconds) of scoreboard occlusion to be registered as absent
    'min_absence_duration': get_env_float('MIN_ABSENCE_DURATION', 0.5),
    
    # Minimum spatial gradient Sobel score for HUD detection stability filter
    'hud_grad_min': get_env_float('HUD_GRAD_MIN', 20.0),
    
    # Number of validation frames to exclude from calibration to test read performance
    'validation_holdout_count': get_env_int('VALIDATION_HOLDOUT_COUNT', 10),
    
    # Minimum scoreboard localized width fraction relative to frame width
    'scoreboard_min_width_frac': get_env_float('SCOREBOARD_MIN_WIDTH_FRAC', 0.05),
    
    # Horizontal structuring element width ratio used for morphological dilation
    'morph_close_w_frac': get_env_float('MORPH_CLOSE_W_FRAC', 0.04),
    
    # Minimum horizontal morphological structure element width (in pixels)
    'morph_close_min_w': get_env_int('MORPH_CLOSE_MIN_W', 25),
    
    # Bounding box IoU threshold for consensus score bounding box clustering
    'ocr_consensus_iou_threshold': get_env_float('OCR_CONSENSUS_IOU_THRESHOLD', 0.4),
    
    # Maximum character box width fraction relative to scoreboard
    'max_char_width_frac': get_env_float('MAX_CHAR_WIDTH_FRAC', 0.25),
    
    # Maximum character box height fraction relative to scoreboard
    'max_char_height_frac': get_env_float('MAX_CHAR_HEIGHT_FRAC', 0.80),
    
    # Digit bounding box padding fraction applied during crop refining
    'digit_bbox_pad_frac': get_env_float('DIGIT_BBOX_PAD_FRAC', 0.15),
    
    # Minimum digit contour height ratio during refinement
    'refine_contour_h_min_frac': get_env_float('REFINE_CONTOUR_H_MIN_FRAC', 0.15),
    
    # Maximum digit contour height ratio during refinement
    'refine_contour_h_max_frac': get_env_float('REFINE_CONTOUR_H_MAX_FRAC', 0.95),
    
    # Minimum digit contour width ratio during refinement
    'refine_contour_w_min_frac': get_env_float('REFINE_CONTOUR_W_MIN_FRAC', 0.10),
    
    # Maximum digit contour width ratio during refinement
    'refine_contour_w_max_frac': get_env_float('REFINE_CONTOUR_W_MAX_FRAC', 0.95),
    
    # Mean pixel diff threshold to assume score change cache hit (avoiding OCR calls)
    'change_cache_threshold': get_env_float('CHANGE_CACHE_THRESHOLD', 3.5),
    
    # Median MAD multiplier to dynamically calibrate scoreboard presence detection limits
    'static_mask_mad_multiplier': get_env_float('STATIC_MASK_MAD_MULTIPLIER', 2),

    # ==========================================================================
    # Token-based OCR score/timer parsing heuristics
    # ==========================================================================

    # Maximum valid minutes value for a football match clock (timer parsing)
    'timer_max_minutes': get_env_int('TIMER_MAX_MINUTES', 130),

    # Maximum valid seconds value for a football match clock (timer parsing)
    'timer_max_seconds': get_env_int('TIMER_MAX_SECONDS', 59),

    # Fraction of scoreboard width used as the allowed horizontal gap around a ':' in timer assembly
    'timer_colon_gap_frac': get_env_float('TIMER_COLON_GAP_FRAC', 0.18),

    # Minimum allowed horizontal gap (pixels) around a ':' in timer assembly
    'timer_colon_gap_min_px': get_env_int('TIMER_COLON_GAP_MIN_PX', 20),

    # Minimum vertical-overlap ratio required for timer digits to be on the same row as ':'
    'timer_voverlap_min': get_env_float('TIMER_VOVERLAP_MIN', 0.45),

    # Maximum number of best candidate digit pairs tested per colon token during timer assembly
    'timer_candidate_pairs': get_env_int('TIMER_CANDIDATE_PAIRS', 4),

    # Weight applied to vertical-alignment score when ranking assembled timer candidates
    'timer_row_score_w': get_env_float('TIMER_ROW_SCORE_W', 15.0),

    # Weight applied to horizontal gap penalty when ranking assembled timer candidates
    'timer_gap_penalty_w': get_env_float('TIMER_GAP_PENALTY_W', 0.8),

    # Maximum vertical row-tolerance (fraction of token height) for pairing split score tokens
    'score_row_tol': get_env_float('SCORE_ROW_TOL', 0.35),

    # Minimum horizontal separation (fraction of scoreboard width) between split score tokens
    'score_split_sep_frac': get_env_float('SCORE_SPLIT_SEP_FRAC', 0.10),

    # Bonus score awarded when a split-score pair straddles the scoreboard horizontal center
    'score_straddle_bonus': get_env_float('SCORE_STRADDLE_BONUS', 25.0),

    # Penalty weight applied to distance from scoreboard center when ranking split-score pairs
    'score_center_penalty_w': get_env_float('SCORE_CENTER_PENALTY_W', 1.2),

    # Penalty weight applied to vertical misalignment when ranking split-score pairs
    'score_y_penalty_w': get_env_float('SCORE_Y_PENALTY_W', 0.2),

    # Maximum width fraction (of scoreboard) allowed for a digit token to be considered a score digit
    'digit_max_w_frac': get_env_float('DIGIT_MAX_W_FRAC', 0.25),

    # Maximum height fraction (of scoreboard) allowed for a digit token to be considered a score digit
    'digit_max_h_frac': get_env_float('DIGIT_MAX_H_FRAC', 0.85),

    # Maximum IOU overlap allowed between a candidate score digit and the timer before rejection
    'timer_overlap_reject': get_env_float('TIMER_OVERLAP_REJECT', 0.25),
}

# ==============================================================================
# Replay Detector Default Configuration Parameters
# ==============================================================================
REPLAY_DETECTOR_DEFAULT_CONFIG = {
    # Frame window size on each side of the sliding detector to compute sustained non-green shifts
    'step_window_sec': get_env_float('STEP_WINDOW_SEC', 2.0),
    
    # Minimum sustained non-green transition level shift required to trigger a wipe candidate
    'step_th': get_env_float('STEP_TH', 0.23),
    
    # Green ratio threshold at which we stop expanding detected transition segments
    'green_th': get_env_float('GREEN_TH', 0.70),
    
    # Number of frames to expand the search window to the left of the anchor
    'search_left_offset': get_env_int('SEARCH_LEFT_OFFSET', 50),
    
    # Number of frames to expand the search window to the right of the anchor
    'search_right_offset': get_env_int('SEARCH_RIGHT_OFFSET', 15),
    
    # Maximum distance (in frames) between two wipe events to merge them into a single segment
    'wipe_merge_frames': get_env_int('WIPE_MERGE_FRAMES', 30),
    
    # Seconds before goal anchor to search for the replay entrance transitions
    'replay_search_buffer': get_env_float('REPLAY_SEARCH_BUFFER', 60.0),
    
    # Seconds after goal anchor to scan for replay exit transitions
    'replay_search_end_buffer': get_env_float('REPLAY_SEARCH_END_BUFFER', 60.0),
    
    # Minimum seconds of scoreboard occlusion to register a valid absent segment
    'min_absence_duration': get_env_float('MIN_ABSENCE_DURATION', 0.5),
    
    # Minimum duration (seconds) of scoreboard absence needed to override visual wipe boundaries
    'min_absence_priority_duration': get_env_float('MIN_ABSENCE_PRIORITY_DURATION', 10.0),
    
    # Weight score contribution of the product signal (block-max x drop)
    'weight_product': get_env_float('WEIGHT_PRODUCT', 1.0),
    
    # Weight score contribution of the block max non-green delta signal
    'weight_block_max': get_env_float('WEIGHT_BLOCK_MAX', 1.0),
    
    # Weight score contribution of the first derivative signal
    'weight_first_deriv': get_env_float('WEIGHT_FIRST_DERIV', 1.0),
    
    # Weight score contribution of the inverse step-shift signal
    'weight_inverse_step': get_env_float('WEIGHT_INVERSE_STEP', 1.5),
    
    # Enable/disable using product signal for wipe pairing scores
    'use_product': get_env_bool('USE_PRODUCT', True),
    
    # Enable/disable using block max signal for wipe pairing scores
    'use_block_max': get_env_bool('USE_BLOCK_MAX', False),
    
    # Enable/disable using first derivative signal for wipe pairing scores
    'use_first_deriv': get_env_bool('USE_FIRST_DERIV', False),
    
    # Enable/disable using inverse step-shift signal for wipe pairing scores
    'use_inverse_step': get_env_bool('USE_INVERSE_STEP', False),
    
    # Enable/disable normalizing wipe signals prior to pairing calculations
    'normalize_signals': get_env_bool('NORMALIZE_SIGNALS', False),
    
    # Forward search window (seconds) to locate exit wipe candidates from the entrance peak
    'peak_forward_window_sec': get_env_float('PEAK_FORWARD_WINDOW_SEC', 17.0),
    
    # Seconds the non-green ratio drop must remain low to consider the HUD stably absent
    'drop_stay_sec': get_env_float('DROP_STAY_SEC', 4.0),
    
    # Threshold tolerance level added to the baseline non-green ratio for drop detection
    'drop_tolerance': get_env_float('DROP_TOLERANCE', 0.1),
    
    # Grass green color classification range lower bound (HSV)
    'green_hsv_lower': get_env_list('GREEN_HSV_LOWER', [35, 40, 40]),
    
    # Grass green color classification range upper bound (HSV)
    'green_hsv_upper': get_env_list('GREEN_HSV_UPPER', [85, 255, 255]),
    
    # Jersey blue color classification range lower bound (HSV)
    'blue_hsv_lower': get_env_list('BLUE_HSV_LOWER', [100, 50, 50]),
    
    # Jersey blue color classification range upper bound (HSV)
    'blue_hsv_upper': get_env_list('BLUE_HSV_UPPER', [140, 255, 255]),
    
    # Jersey red color classification range lower bound 1 (HSV)
    'red_hsv_lower1': get_env_list('RED_HSV_LOWER1', [0, 50, 50]),
    
    # Jersey red color classification range upper bound 1 (HSV)
    'red_hsv_upper1': get_env_list('RED_HSV_UPPER1', [10, 255, 255]),
    
    # Jersey red color classification range lower bound 2 (HSV)
    'red_hsv_lower2': get_env_list('RED_HSV_LOWER2', [170, 50, 50]),
    
    # Jersey red color classification range upper bound 2 (HSV)
    'red_hsv_upper2': get_env_list('RED_HSV_UPPER2', [180, 255, 255]),
    
    # Line white color classification range lower bound (HSV)
    'white_hsv_lower': get_env_list('WHITE_HSV_LOWER', [0, 0, 220]),
    
    # Line white color classification range upper bound (HSV)
    'white_hsv_upper': get_env_list('WHITE_HSV_UPPER', [180, 30, 255]),
    
    # Minimum replay clip duration (seconds) allowed for pairing
    'min_replay_duration': get_env_float('MIN_REPLAY_DURATION', 19.0),
    
    # Maximum replay clip duration (seconds) allowed for pairing
    'max_replay_duration': get_env_float('MAX_REPLAY_DURATION', 70.0),
    
    # Optimal target minimum duration (seconds) for football replays
    'optimal_replay_duration_min': get_env_float('OPTIMAL_REPLAY_DURATION_MIN', 20.0),
    
    # Optimal target maximum duration (seconds) for football replays
    'optimal_replay_duration_max': get_env_float('OPTIMAL_REPLAY_DURATION_MAX', 30.0),
    
    # Reference short duration reference used to apply duration penalties
    'duration_penalty_short_ref': get_env_float('DURATION_PENALTY_SHORT_REF', 20.0),
    
    # Reference long duration reference used to apply duration penalties
    'duration_penalty_long_ref': get_env_float('DURATION_PENALTY_LONG_REF', 30.0),
    
    # Lower bound duration (seconds) under which short duration penalties escalate
    'duration_penalty_short_tol': get_env_float('DURATION_PENALTY_SHORT_TOL', 25.0),
    
    # Penalty multiplier for clips shorter than the optimal range
    'duration_penalty_short_multiplier': get_env_float('DURATION_PENALTY_SHORT_MULTIPLIER', 0.5),
    
    # Penalty multiplier for clips longer than the optimal range
    'duration_penalty_long_multiplier': get_env_float('DURATION_PENALTY_LONG_MULTIPLIER', 0.05),
    
    # Scoreboard absence duration threshold (seconds) needed to be eligible for overlap bonus
    'absence_bonus_duration_th': get_env_float('ABSENCE_BONUS_DURATION_TH', 20.0),
    
    # Minimum overlap duration (seconds) between wipe segment and absent HUD to get full bonus
    'absence_bonus_min_overlap': get_env_float('ABSENCE_BONUS_MIN_OVERLAP', 5.0),
    
    # Ratio of replay duration that must overlap with HUD absence to get full bonus
    'absence_bonus_overlap_ratio': get_env_float('ABSENCE_BONUS_OVERLAP_RATIO', 0.5),
    
    # Score bonus awarded for strong scoreboard absence alignment
    'absence_bonus_max_val': get_env_float('ABSENCE_BONUS_MAX_VAL', 1.0),
    
    # Score bonus awarded for partial scoreboard absence alignment
    'absence_bonus_mid_val': get_env_float('ABSENCE_BONUS_MID_VAL', 0.3),
    
    # Weight of the anchor delay penalty (deducted per second of delay after goal)
    'anchor_delay_penalty_weight': get_env_float('ANCHOR_DELAY_PENALTY_WEIGHT', 0.15),
    
    # Fallback replay clip length (seconds) if entrance/exit wipe transitions aren't found
    'fallback_replay_duration': get_env_float('FALLBACK_REPLAY_DURATION', 25.0),
}

# ==============================================================================
# Worker Service Default Configuration Parameters
# ==============================================================================
WORKER_DEFAULT_CONFIG = {
    'kafka_bootstrap_servers': os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
    'kafka_consumer_group_id': os.environ.get('KAFKA_CONSUMER_GROUP_ID', 'ocr-detector-group'),
    'kafka_task_topic': os.environ.get('KAFKA_TASK_TOPIC', 'ocr-play-clusters'),
    'max_workers': get_env_int('MAX_WORKERS', 2),
    'processing_timeout_seconds': get_env_int('PROCESSING_TIMEOUT_SECONDS', 750), 
    'encrypt_key': os.environ.get('ENCRYPT_KEY', 'tv360'),
    'worker_poll_interval_sec': get_env_float('WORKER_POLL_INTERVAL_SEC', 5.0),
    # MySQL Database configurations
    'mysql_host': os.environ.get('MYSQL_HOST', '127.0.0.1'),
    'mysql_port': get_env_int('MYSQL_PORT', 3306),
    'mysql_user': os.environ.get('MYSQL_USER', 'clipper_user'),
    'mysql_password': os.environ.get('MYSQL_PASSWORD', 'clipper_pass'),
    'mysql_database': os.environ.get('MYSQL_DATABASE', 'live_clipper'),
}

