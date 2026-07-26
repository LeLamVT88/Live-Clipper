"""CSV schemas produced by lineup resolution."""

RESOLVED_COLUMNS = [
    "video",
    "segment_index",
    "lineup_index",
    "resolution_method",
    "formation_timestamp_seconds",
    "slot_index",
    "row_index",
    "shirt_number",
    "formation_label",
    "player_name",
    "number_confidence",
    "label_confidence",
    "name_confidence",
    "pair_confidence",
    "number_evidence_frames",
    "label_evidence_frames",
    "full_name_evidence_frames",
    "slot_center_x_norm",
    "slot_center_y_norm",
]

DIAGNOSTIC_COLUMNS = [
    "video",
    "segment_index",
    "status",
    "resolution_method",
    "resolved_players",
    "message",
]
