"""Scene-level OCR primitives for transcript-free lineup detection."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .schema import LineupDetectionError
from .utils import PROJECT_ROOT


DEFAULT_UNIT_SECONDS = 3.0
MIN_CONFIDENT_LINEUP_DURATION_SECONDS = 8.0
DEFAULT_OCR_PYTHON = PROJECT_ROOT / ".venv-ocr" / "bin" / "python"
TEAM_LABEL_EXCLUSIONS = frozenset(
    {
        "COACH",
        "FORMATION",
        "FIFA",
        "GK",
        "LINEUP",
        "STARTING XI",
        "SUBSTITUTES",
    }
)

SceneDetector = Callable[..., tuple[float, ...]]
OCRRunner = Callable[..., dict[str, object]]


@dataclass(frozen=True)
class VisualUnit:
    """One visually stable interval represented by its middle frame."""

    start_seconds: float
    end_seconds: float
    scene_index: int

    @property
    def center_seconds(self) -> float:
        return (self.start_seconds + self.end_seconds) / 2

    def to_dict(self) -> dict[str, float | int]:
        return {
            "start_seconds": round(self.start_seconds, 3),
            "end_seconds": round(self.end_seconds, 3),
            "sample_seconds": round(self.center_seconds, 3),
            "scene_index": self.scene_index,
        }


@dataclass(frozen=True)
class VisualSearchTask:
    task_id: str
    mode: str
    center_seconds: float
    max_events: int
    units: tuple[VisualUnit, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "mode": self.mode,
            "center_seconds": round(self.center_seconds, 3),
            "max_events": self.max_events,
            "units": [unit.to_dict() for unit in self.units],
        }


@dataclass(frozen=True)
class VisualEvent:
    start_seconds: float
    end_seconds: float
    confidence: float
    task_id: str
    sample_seconds: tuple[float, ...] = ()
    texts: tuple[str, ...] = ()

    @property
    def center_seconds(self) -> float:
        return (self.start_seconds + self.end_seconds) / 2

    def to_dict(self) -> dict[str, object]:
        return {
            "start_seconds": round(self.start_seconds, 3),
            "end_seconds": round(self.end_seconds, 3),
            "confidence": round(self.confidence, 4),
            "task_id": self.task_id,
            "sample_seconds": [round(value, 3) for value in self.sample_seconds],
            "texts": list(self.texts),
        }


def build_visual_units(
    *,
    start_seconds: float,
    end_seconds: float,
    scene_cuts: Iterable[float],
    max_unit_seconds: float = DEFAULT_UNIT_SECONDS,
) -> tuple[VisualUnit, ...]:
    """Split long scenes so representative frames never have a large gap."""

    if not all(math.isfinite(value) for value in (start_seconds, end_seconds)):
        raise LineupDetectionError("Visual search bounds must be finite.")
    if start_seconds < 0 or end_seconds <= start_seconds:
        raise LineupDetectionError("Invalid visual search range.")
    if not math.isfinite(max_unit_seconds) or max_unit_seconds <= 0:
        raise LineupDetectionError("Visual unit duration must be positive.")
    boundaries = [start_seconds]
    boundaries.extend(
        sorted(
            {
                float(cut)
                for cut in scene_cuts
                if math.isfinite(float(cut))
                and start_seconds < float(cut) < end_seconds
            }
        )
    )
    boundaries.append(end_seconds)

    units: list[VisualUnit] = []
    for scene_index, (scene_start, scene_end) in enumerate(
        zip(boundaries, boundaries[1:])
    ):
        part_count = max(1, math.ceil((scene_end - scene_start) / max_unit_seconds))
        part_duration = (scene_end - scene_start) / part_count
        for part_index in range(part_count):
            unit_start = scene_start + part_index * part_duration
            unit_end = (
                scene_end
                if part_index == part_count - 1
                else scene_start + (part_index + 1) * part_duration
            )
            units.append(VisualUnit(unit_start, unit_end, scene_index))
    return tuple(units)


def build_global_task(
    *,
    scene_cuts: Iterable[float],
    scan_start_seconds: float,
    scan_end_seconds: float,
    max_events: int = 2,
    max_unit_seconds: float = DEFAULT_UNIT_SECONDS,
) -> VisualSearchTask:
    return VisualSearchTask(
        task_id="global_scan",
        mode="global",
        center_seconds=(scan_start_seconds + scan_end_seconds) / 2,
        max_events=max_events,
        units=build_visual_units(
            start_seconds=scan_start_seconds,
            end_seconds=scan_end_seconds,
            scene_cuts=scene_cuts,
            max_unit_seconds=max_unit_seconds,
        ),
    )


def run_ocr_worker(
    video_path: Path,
    tasks: Sequence[VisualSearchTask],
    *,
    python_path: Path = DEFAULT_OCR_PYTHON,
    max_recognition_frames: int = 40,
) -> dict[str, object]:
    """Run PaddleOCR in its dedicated Python environment."""

    if not tasks:
        return {"tasks": [], "model_loaded": False}
    executable = python_path.expanduser()
    if not executable.is_absolute():
        executable = PROJECT_ROOT / executable
    if not executable.is_file():
        raise LineupDetectionError(
            "OCR Python does not exist: "
            f"{executable}. Create .venv-ocr or pass --ocr-python."
        )
    request = {
        "video_path": str(video_path.resolve()),
        "max_recognition_frames": max_recognition_frames,
        "tasks": [task.to_dict() for task in tasks],
    }
    worker_path = Path(__file__).with_name("ocr_worker.py")
    with tempfile.TemporaryDirectory(prefix="lineup_ocr_") as directory:
        request_path = Path(directory) / "request.json"
        output_path = Path(directory) / "response.json"
        request_path.write_text(
            json.dumps(request, ensure_ascii=False), encoding="utf-8"
        )
        environment = dict(os.environ)
        environment.setdefault(
            "PADDLE_PDX_CACHE_HOME", str(PROJECT_ROOT / ".cache" / "paddlex")
        )
        environment.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        completed = subprocess.run(
            [
                str(executable),
                str(worker_path),
                "--request",
                str(request_path),
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
        if completed.returncode != 0 or not output_path.is_file():
            detail = (completed.stderr or completed.stdout).strip()[-1500:]
            raise LineupDetectionError(
                f"Lineup OCR worker failed ({completed.returncode}): {detail}"
            )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise LineupDetectionError("Lineup OCR worker returned invalid JSON.")
    return payload


def events_from_worker(
    payload: Mapping[str, object],
) -> dict[str, tuple[VisualEvent, ...]]:
    """Validate the compact worker response at the process boundary."""

    parsed: dict[str, tuple[VisualEvent, ...]] = {}
    raw_tasks = payload.get("tasks", [])
    if not isinstance(raw_tasks, list):
        raise LineupDetectionError("Lineup OCR tasks must be an array.")
    for raw_task in raw_tasks:
        if not isinstance(raw_task, dict):
            raise LineupDetectionError("Invalid lineup OCR task result.")
        task_id = str(raw_task.get("task_id", "")).strip()
        raw_events = raw_task.get("events", [])
        if not task_id or not isinstance(raw_events, list):
            raise LineupDetectionError("Invalid lineup OCR task result.")
        events: list[VisualEvent] = []
        for raw_event in raw_events:
            if not isinstance(raw_event, dict):
                raise LineupDetectionError("Invalid lineup OCR event.")
            try:
                event = VisualEvent(
                    start_seconds=float(raw_event["start_seconds"]),
                    end_seconds=float(raw_event["end_seconds"]),
                    confidence=float(raw_event["confidence"]),
                    task_id=task_id,
                    sample_seconds=tuple(
                        float(value)
                        for value in raw_event.get("sample_seconds", [])
                    ),
                    texts=tuple(str(value) for value in raw_event.get("texts", [])),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise LineupDetectionError("Invalid lineup OCR event values.") from exc
            if (
                event.start_seconds < 0
                or event.end_seconds <= event.start_seconds
                or not 0 <= event.confidence <= 1
            ):
                raise LineupDetectionError("Invalid lineup OCR event range.")
            events.append(event)
        parsed[task_id] = tuple(events)
    return parsed


def snap_visual_event(
    event: VisualEvent,
    scene_cuts: Iterable[float],
    *,
    radius_seconds: float,
) -> VisualEvent:
    """Snap only outward so scene alignment never trims visual evidence."""

    cuts = tuple(
        sorted(
            float(cut)
            for cut in scene_cuts
            if math.isfinite(float(cut)) and float(cut) >= 0
        )
    )
    earlier_starts = tuple(
        cut
        for cut in cuts
        if 1e-6 < event.start_seconds - cut <= radius_seconds
    )
    later_ends = tuple(
        cut
        for cut in cuts
        if -1e-3 <= cut - event.end_seconds <= radius_seconds
    )
    start = max(earlier_starts, default=event.start_seconds)
    end = min(later_ends, default=event.end_seconds)
    if end <= start + 0.25:
        return event
    return replace(event, start_seconds=start, end_seconds=end)


def _normalized_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).casefold()
    return " ".join(
        "".join(
            character
            for character in normalized
            if not unicodedata.combining(character)
            and (character.isalnum() or character.isspace())
        ).split()
    )


def visual_team_hint(event: VisualEvent) -> str | None:
    """Read a likely uppercase team title without requiring player names."""

    for raw_text in event.texts[:4]:
        text = " ".join(raw_text.strip().split())
        normalized = _normalized_text(text).upper()
        letters = "".join(character for character in text if character.isalpha())
        if (
            3 <= len(text) <= 40
            and letters
            and text == text.upper()
            and normalized not in TEAM_LABEL_EXCLUSIONS
            and all(character.isalpha() or character in " -.'" for character in text)
        ):
            return text
    return None
