"""Orchestrate table, formation, and targeted local OCR resolution."""

from __future__ import annotations

import pandas as pd

from .common import LineupResolutionError
from .formation import attempt_formation_resolution
from .formation_refinement import refine_formation_numbers
from .layout import formation_refinement_frames
from .local_models import (
    create_local_number_recognizer,
    create_local_table_ocr,
)
from .table import resolve_table_layout
from .table_refinement import refine_table_numbers


def resolve_all_lineups(
    detections: pd.DataFrame,
    expected_players: int,
    min_number_count: int,
    max_gap_seconds: float,
    signature_threshold: float,
    enable_local_ocr: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    local_ocr: object | None = None
    local_number_recognizer: object | None = None

    grouped = detections.groupby(
        ["video", "segment_index"],
        sort=False,
    )
    for (video, segment_index), segment in grouped:
        messages: list[str] = []
        table_records, table_count = resolve_table_layout(
            segment,
            expected_players=expected_players,
        )
        if table_records:
            records.extend(table_records)
            diagnostics.append(
                {
                    "video": video,
                    "segment_index": int(segment_index),
                    "status": "resolved",
                    "resolution_method": "table",
                    "resolved_players": len(table_records),
                    "message": (
                        "Complete repeated table/list consensus."
                    ),
                }
            )
            print(
                f"{video}, segment {int(segment_index)}: "
                f"{len(table_records)} players resolved "
                f"from table/list"
            )
            continue

        if table_count:
            messages.append(
                f"table/list pass found {table_count}/"
                f"{expected_players} players"
            )
        if (
            enable_local_ocr
            and 4 <= table_count < expected_players
            and "frame_path" in segment.columns
        ):
            try:
                if local_ocr is None:
                    print(
                        "Loading PaddleOCR for local "
                        "table-number refinement..."
                    )
                    local_ocr = create_local_table_ocr()
                refined_segment, added_count = (
                    refine_table_numbers(
                        segment,
                        ocr=local_ocr,
                        expected_players=expected_players,
                    )
                )
                if added_count:
                    messages.append(
                        f"local table OCR added {added_count} "
                        "number observations"
                    )
                    (
                        table_records,
                        refined_table_count,
                    ) = resolve_table_layout(
                        refined_segment,
                        expected_players=expected_players,
                    )
                    table_count = max(
                        table_count,
                        refined_table_count,
                    )
                    segment = refined_segment
                if table_records:
                    for record in table_records:
                        record["resolution_method"] = (
                            "table+local_ocr"
                        )
                    records.extend(table_records)
                    diagnostics.append(
                        {
                            "video": video,
                            "segment_index": int(
                                segment_index
                            ),
                            "status": "resolved",
                            "resolution_method": (
                                "table+local_ocr"
                            ),
                            "resolved_players": len(
                                table_records
                            ),
                            "message": "; ".join(messages),
                        }
                    )
                    print(
                        f"{video}, segment "
                        f"{int(segment_index)}: "
                        f"{len(table_records)} players resolved "
                        f"from table/list after local OCR"
                    )
                    continue
            except (
                ImportError,
                LineupResolutionError,
                ModuleNotFoundError,
                OSError,
                ValueError,
            ) as exc:
                messages.append(
                    f"local table OCR unavailable: {exc}"
                )

        (
            _,
            initial_events,
            initial_records,
            initial_errors,
        ) = attempt_formation_resolution(
            segment,
            expected_players=expected_players,
            min_number_count=min_number_count,
            max_gap_seconds=max_gap_seconds,
            signature_threshold=signature_threshold,
        )
        if initial_events and not initial_errors:
            records.extend(initial_records)
            diagnostics.append(
                {
                    "video": video,
                    "segment_index": int(segment_index),
                    "status": "resolved",
                    "resolution_method": "formation",
                    "resolved_players": len(initial_records),
                    "message": "; ".join(messages),
                }
            )
            print(
                f"{video}, segment {int(segment_index)}: "
                f"{len(initial_events)} formation(s), "
                f"{len(initial_records)} players resolved"
            )
            continue

        if (
            enable_local_ocr
            and "frame_path" in segment.columns
            and formation_refinement_frames(segment)
        ):
            try:
                if local_ocr is None:
                    print(
                        "Loading PaddleOCR for local "
                        "formation-number refinement..."
                    )
                    local_ocr = create_local_table_ocr()
                if local_number_recognizer is None:
                    local_number_recognizer = (
                        create_local_number_recognizer()
                    )
                refined_segment, added_count = (
                    refine_formation_numbers(
                        segment,
                        ocr=local_ocr,
                        recognizer=local_number_recognizer,
                    )
                )
                if added_count:
                    segment = refined_segment
                    messages.append(
                        f"local formation OCR added "
                        f"{added_count} number observations"
                    )
            except (
                ImportError,
                LineupResolutionError,
                ModuleNotFoundError,
                OSError,
                ValueError,
            ) as exc:
                messages.append(
                    f"local formation OCR unavailable: {exc}"
                )

        (
            _,
            events,
            segment_records,
            event_errors,
        ) = attempt_formation_resolution(
            segment,
            expected_players=expected_players,
            min_number_count=min_number_count,
            max_gap_seconds=max_gap_seconds,
            signature_threshold=signature_threshold,
        )
        if not events:
            messages.append("no formation snapshot found")
            diagnostics.append(
                {
                    "video": video,
                    "segment_index": int(segment_index),
                    "status": "unresolved",
                    "resolution_method": "",
                    "resolved_players": 0,
                    "message": "; ".join(messages),
                }
            )
            print(
                f"{video}, segment {int(segment_index)}: "
                f"unresolved ({'; '.join(messages)})"
            )
            continue

        print(
            f"{video}, segment {int(segment_index)}: "
            f"detected {len(events)} lineup formation(s)"
        )
        if segment_records:
            print(
                f"  resolved {len(segment_records)} player rows"
            )
        for error in event_errors:
            label = (
                "ignored incomplete candidate"
                if segment_records
                else "unresolved"
            )
            print(f"  {label} ({error})")

        resolution_method = (
            "formation+local_ocr"
            if any(
                message.startswith(
                    "local formation OCR added"
                )
                for message in messages
            )
            else "formation"
        )
        for record in segment_records:
            record["resolution_method"] = resolution_method
        records.extend(segment_records)
        messages.extend(
            (
                "ignored incomplete formation candidate: "
                f"{error}"
                if segment_records
                else error
            )
            for error in event_errors
        )
        diagnostics.append(
            {
                "video": video,
                "segment_index": int(segment_index),
                "status": (
                    "resolved"
                    if segment_records
                    else "unresolved"
                ),
                "resolution_method": (
                    resolution_method
                    if segment_records
                    else ""
                ),
                "resolved_players": len(segment_records),
                "message": "; ".join(messages),
            }
        )

    return records, diagnostics
