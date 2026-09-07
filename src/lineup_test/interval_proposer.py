from __future__ import annotations

from typing import Sequence

import numpy as np

from lineup_test.config import PipelineConfig
from lineup_test.schema import LineupInterval


def propose_lineups(
    timestamps: list[float],
    scores: list[float],
    scenes: list[tuple[float, float]],
    config: PipelineConfig | None = None,
    *,
    strong_flags: Sequence[bool] | None = None,
) -> list[LineupInterval]:
    """Build persistent, structurally supported lineup intervals from OCR samples."""
    cfg = config or PipelineConfig()
    if not scores:
        return []
    if len(timestamps) != len(scores):
        raise ValueError("timestamps and scores must have the same length")
    if strong_flags is None:
        evidence_flags = [score >= cfg.strong_score_threshold for score in scores]
    else:
        if len(strong_flags) != len(scores):
            raise ValueError("strong_flags and scores must have the same length")
        evidence_flags = list(strong_flags)

    positive_indices = [
        index for index, score in enumerate(scores) if score >= cfg.decision_threshold
    ]
    if not positive_indices:
        # Only structurally strong samples may use the recall-oriented fallback.
        positive_indices = [
            index
            for index, score in enumerate(scores)
            if score >= cfg.fallback_decision_threshold and evidence_flags[index]
        ]
        if not positive_indices:
            return []

    def in_same_scene(t1: float, t2: float) -> bool:
        return any(s <= t1 <= e and s <= t2 <= e for s, e in scenes)

    # 1. Cluster positive samples, but let a clear negative sample break a chain.
    # This prevents unrelated pre-match text from being joined transitively to a
    # later lineup just because every positive-to-positive gap is under 12s.
    clusters: list[list[int]] = [[positive_indices[0]]]
    for index in positive_indices[1:]:
        previous_index = clusters[-1][-1]
        previous_time = timestamps[previous_index]
        current_time = timestamps[index]
        has_negative_barrier = any(
            scores[middle] <= cfg.cluster_break_score
            for middle in range(previous_index + 1, index)
        )
        close_enough = (
            in_same_scene(previous_time, current_time)
            or current_time - previous_time <= cfg.intra_scene_cluster_gap_sec
        )
        if close_enough and not has_negative_barrier:
            clusters[-1].append(index)
        else:
            clusters.append([index])

    raw_proposals: list[dict[str, object]] = []
    for cluster_indices in clusters:
        strong_sample_count = sum(evidence_flags[index] for index in cluster_indices)
        if (
            len(cluster_indices) < cfg.min_positive_samples
            or strong_sample_count < cfg.min_strong_samples
        ):
            continue

        c_start = timestamps[cluster_indices[0]]
        c_end = timestamps[cluster_indices[-1]]
        # Rank from supporting samples only. Negative frames are used as cluster
        # boundaries and must not dilute a valid lineup's score.
        support_scores = [scores[index] for index in cluster_indices]
        avg_score = float(np.mean(support_scores))

        # Snap outward to matching scenes
        overlapping_scenes = [
            (s, e) for s, e in scenes if s <= c_end and e >= c_start
        ]
        if overlapping_scenes:
            snap_s = min(s for s, e in overlapping_scenes)
            snap_e = max(e for s, e in overlapping_scenes)
            # If scene is excessively long (>40s) and graphic started inside it, bound snap
            if snap_e - snap_s > 40.0:
                snap_s = max(snap_s, c_start - 12.0)
                snap_e = min(snap_e, c_end + 8.0)
        else:
            snap_s = max(0.0, c_start - 4.0)
            snap_e = c_end + 4.0

        duration = snap_e - snap_s
        # Temporal persistence verification: starting lineups must persist for at least min_duration
        if duration >= cfg.min_lineup_duration_sec:
            rank_score = avg_score * min(1.0, duration / 15.0)
            raw_proposals.append(
                {
                    "start": round(snap_s, 2),
                    "end": round(snap_e, 2),
                    "confidence": round(rank_score, 4),
                    "sample_count": len(cluster_indices),
                    "strong_sample_count": strong_sample_count,
                }
            )

    # 2. Merge contiguous proposed spans belonging to the same lineup (gap <= inter_proposal_merge_gap_sec)
    # e.g., multi-shot lineups across adjacent camera cuts (like Leipzig 120-156s and 156-171s)
    merged_proposals: list[dict[str, object]] = []
    proposals_by_time = sorted(raw_proposals, key=lambda x: x["start"])
    for cand in proposals_by_time:
        if not merged_proposals:
            merged_proposals.append(cand)
        else:
            prev = merged_proposals[-1]
            if cand["start"] <= prev["end"] + cfg.inter_proposal_merge_gap_sec:
                prev["end"] = max(prev["end"], cand["end"])
                prev["confidence"] = max(prev["confidence"], cand["confidence"])
                prev["sample_count"] += cand["sample_count"]
                prev["strong_sample_count"] += cand["strong_sample_count"]
                prev["merged_adjacent"] = True
            else:
                merged_proposals.append(cand)

    # Sort merged proposals by score
    merged_proposals.sort(key=lambda x: -x["confidence"])

    # 3. Non-Maximum Suppression (NMS): select top distinct lineup intervals
    selected: list[dict[str, object]] = []
    for cand in merged_proposals:
        s1, e1 = cand["start"], cand["end"]
        overlap = False
        for chosen in selected:
            s2, e2 = chosen["start"], chosen["end"]
            inter = max(0.0, min(e1, e2) - max(s1, s2))
            union = max(e1, e2) - min(s1, s2)
            iou = inter / union if union > 0 else 0.0
            if iou > cfg.nms_iou_threshold:
                overlap = True
                break
        if not overlap:
            selected.append(cand)
            if len(selected) == cfg.max_lineup_intervals:
                break

    selected.sort(key=lambda x: x["start"])

    # 4. Adjacent lineup merge (user requirement):
    # If 2 selected starting lineups are shown back-to-back with minimal transition (gap <= contiguous_lineup_merge_gap_sec,
    # e.g. Bundesliga Freiburg vs Leipzig), merge them into 1 unified lineup output.
    if len(selected) == 2:
        gap = selected[1]["start"] - selected[0]["end"]
        if 0.0 <= gap <= cfg.contiguous_lineup_merge_gap_sec:
            merged_start = selected[0]["start"]
            merged_end = selected[1]["end"]
            merged_score = round((selected[0]["confidence"] + selected[1]["confidence"]) / 2.0, 4)
            merged_count = selected[0]["sample_count"] + selected[1]["sample_count"]
            selected = [
                {
                    "start": merged_start,
                    "end": merged_end,
                    "confidence": merged_score,
                    "sample_count": merged_count,
                    "merged_adjacent": True,
                }
            ]

    # Convert to LineupInterval objects
    result_intervals = [
        LineupInterval(
            start_seconds=item["start"],
            end_seconds=item["end"],
            confidence=item["confidence"],
            sample_count=item.get("sample_count", 0),
            metadata={"merged_adjacent": item.get("merged_adjacent", False)},
        )
        for item in selected
    ]
    return result_intervals
