from __future__ import annotations

import logging
from pathlib import Path
import re

from mapping.config import MappingConfig, load_config
from mapping.csv_export import write_csv
from mapping.ocr import OCRBackend, create_ocr_backend, ocr_player_units
from mapping.player_detection import detect_player_blobs, sort_player_units
from mapping.regions import crop_lineup_region
from mapping.schema import ClipSegment, LineupRow, SquadPlayer, TeamObservation
from mapping.squad import match_against_squad
from mapping.stability import find_stable_segments, select_representative_frame, stable_frames
from mapping.team import detect_team_boundary, read_team_observations, refine_team_boundary
from mapping.video import extract_frames, split_clip, video_duration


LOGGER = logging.getLogger(__name__)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "team"


def _clip_segments(
    duration: float, source: Path, observations: list[TeamObservation], boundaries,
) -> list[ClipSegment]:
    matched = [item.team for item in observations if item.team is not None]
    if not matched:
        return []
    if not boundaries:
        return [ClipSegment(matched[0], 0.0, duration, source)]
    result: list[ClipSegment] = []
    start = 0.0
    current = boundaries[0].from_team
    for boundary in boundaries:
        if boundary.timestamp > start:
            result.append(ClipSegment(current, start, boundary.timestamp, source))
        start, current = boundary.timestamp, boundary.to_team
    if duration > start:
        result.append(ClipSegment(current, start, duration, source))
    return result


def _write_team_clips(
    video_path: Path, segments: list[ClipSegment], split_directory: Path,
) -> list[ClipSegment]:
    if len(segments) <= 1:
        return segments
    result = []
    for index, segment in enumerate(segments, 1):
        output = split_directory / f"{video_path.stem}_{index}_{_safe_name(segment.team)}.mp4"
        split_clip(video_path, segment.start_seconds, segment.end_seconds, output)
        result.append(ClipSegment(
            segment.team, segment.start_seconds, segment.end_seconds, output,
        ))
    return result


def extract_lineup(
    video_path: str | Path,
    teams: list[str],
    *,
    squad: list[SquadPlayer] | None = None,
    config: MappingConfig | None = None,
    config_path: str | Path | None = None,
    output_csv: str | Path | None = None,
    split_directory: str | Path | None = None,
    ocr_backend: OCRBackend | None = None,
) -> list[LineupRow]:
    """Extract starting XI rows for every team graphic found in a short clip."""
    if not teams or any(not team.strip() for team in teams):
        raise ValueError("At least one non-empty team name is required")
    path = Path(video_path)
    cfg = config or load_config(config_path)
    backend = ocr_backend or create_ocr_backend(cfg.ocr)
    duration = video_duration(path)

    frames = extract_frames(path, cfg.sample_interval_sec, end=duration)
    stable = find_stable_segments(frames, cfg.stable_diff_threshold, cfg.stable_min_frames)
    if not stable:
        LOGGER.warning("No stable lineup frame found in %s", path)
        if output_csv is not None:
            write_csv([], output_csv)
        return []

    observations = read_team_observations(stable_frames(frames, stable), teams, cfg, backend)
    matched_teams = list(dict.fromkeys(item.team for item in observations if item.team is not None))
    if not matched_teams:
        LOGGER.warning("Team-name OCR did not match any configured team in %s", path)
        if output_csv is not None:
            write_csv([], output_csv)
        return []

    boundaries = detect_team_boundary(observations, cfg.team_debounce_frames)
    if len(matched_teams) > 1 and not boundaries:
        LOGGER.warning(
            "Both teams were observed but no debounced boundary could be established in %s", path,
        )
    boundaries = [refine_team_boundary(str(path), item, teams, cfg, backend) for item in boundaries]
    segments = _clip_segments(duration, path, observations, boundaries)
    if cfg.write_split_clips and len(segments) > 1:
        destination = Path(split_directory) if split_directory else path.parent / f"{path.stem}_segments"
        segments = _write_team_clips(path, segments, destination)

    result: list[LineupRow] = []
    squad_players = squad or []
    for segment in segments:
        representative = select_representative_frame(
            frames, stable, segment.start_seconds, segment.end_seconds,
        )
        if representative is None:
            LOGGER.warning("No stable representative frame for team %s", segment.team)
            continue
        lineup_image = crop_lineup_region(representative.image, cfg)
        units = detect_player_blobs(lineup_image, cfg.player_detection)
        if len(units) != cfg.expected_starters:
            LOGGER.warning(
                "Detected %d player units for %s; expected %d",
                len(units), segment.team, cfg.expected_starters,
            )
        if len(units) > cfg.expected_starters:
            units = sort_player_units(
                sorted(units, key=lambda item: item.detection_score, reverse=True)[:cfg.expected_starters],
                cfg.player_detection.row_tolerance_ratio,
            )
        raw_players = ocr_player_units(lineup_image, units, backend)
        matched = match_against_squad(
            raw_players, segment.team, squad_players, cfg.squad_similarity_threshold,
        ) if squad_players else [
            (player.jersey_number, player.player_name,
             "high" if player.jersey_number is not None and player.player_name
             and min(player.number_confidence, player.name_confidence) >= .85 else "low")
            for player in raw_players
        ]
        timestamp = representative.timestamp - (segment.start_seconds if segment.source_clip != path else 0.0)
        for raw, (number, name, confidence) in zip(raw_players, matched):
            if not raw.player_name or raw.jersey_number is None:
                LOGGER.warning(
                    "Empty/incomplete player OCR for %s at %.3fs: number=%r name=%r",
                    segment.team, representative.timestamp, raw.jersey_number, raw.player_name,
                )
            if squad_players and confidence == "low":
                LOGGER.warning(
                    "No squad match for %s player OCR: number=%r name=%r",
                    segment.team, raw.jersey_number, raw.player_name,
                )
            result.append(LineupRow(
                segment.team, number, name, "starting", str(segment.source_clip),
                round(timestamp, 3), confidence,
            ))
    if output_csv is not None:
        write_csv(result, output_csv)
    return result
