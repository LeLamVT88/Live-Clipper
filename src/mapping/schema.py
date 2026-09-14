FRAME_COLUMNS = """video segment_index segment_label segment_start_seconds
    segment_end_seconds frame_index frame_path timestamp timestamp_seconds
    relative_seconds frame_width frame_height""".split()
DETECTION_COLUMNS = FRAME_COLUMNS + """text text_type score x1 y1 x2 y2 center_x
    center_y center_x_norm center_y_norm""".split()
SELECTED_FRAME_COLUMNS = FRAME_COLUMNS + """source_frame_path selection_layout
    selection_score scout_frame_index crop_side crop_x1_norm crop_x2_norm""".split()
SELECTION_DIAGNOSTIC_COLUMNS = """video segment_index layout status score scout_frame_index
    scout_timestamp_seconds formation_anchor_count table_pair_count
    number_count name_count crop_side crop_x1_norm crop_x2_norm
    selected_frame_indices message""".split()
ATTEMPT_COLUMNS = """video segment_index attempt tier frame_count new_ocr_frame_count
    detection_count resolver_status resolved_players quality_pass
    quality_message""".split()
RESOLVED_COLUMNS = """video segment_index lineup_index resolution_method
    formation_timestamp_seconds slot_index row_index shirt_number
    formation_label player_name name_source number_source number_confidence label_confidence
    name_confidence pair_confidence number_evidence_frames
    label_evidence_frames full_name_evidence_frames slot_center_x_norm
    slot_center_y_norm""".split()
DIAGNOSTIC_COLUMNS = """video segment_index status resolution_method resolved_players
    message""".split()
