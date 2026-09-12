from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from mapping.schema import Rect


@dataclass(frozen=True, slots=True)
class OCRConfig:
    backend: str = "paddleocr"
    device: str = "cpu"
    detection_model: str = "PP-OCRv6_small_det"
    recognition_model: str = "PP-OCRv6_small_rec"
    upscale_factor: float = 3.0
    llm_api_url: str | None = None
    llm_model: str | None = None
    llm_api_key_env: str = "LINEUP_LLM_API_KEY"

    def __post_init__(self) -> None:
        if self.backend not in {"paddleocr", "llm_vision"}:
            raise ValueError("ocr.backend must be 'paddleocr' or 'llm_vision'")
        if self.upscale_factor < 1:
            raise ValueError("ocr.upscale_factor must be >= 1")


@dataclass(frozen=True, slots=True)
class PlayerDetectionConfig:
    min_saturation: int = 65
    min_value: int = 45
    white_saturation_max: int = 45
    white_value_min: int = 205
    background_color_distance: float = 24.0
    min_area_pct: float = 0.0008
    max_area_pct: float = 0.045
    min_aspect_ratio: float = 0.35
    max_aspect_ratio: float = 2.2
    edge_margin_pct: float = 0.008
    morphology_kernel: int = 5
    name_extension_ratio: float = 0.75
    horizontal_extension_ratio: float = 0.8
    row_tolerance_ratio: float = 0.55


@dataclass(frozen=True, slots=True)
class MappingConfig:
    team_name_bar: Rect
    lineup_region: Rect
    substitutes_region: Rect | None = None
    coach_region: Rect | None = None
    exclude_regions: tuple[Rect, ...] = ()
    sample_interval_sec: float = 0.3
    stable_diff_threshold: float = 2.8
    stable_min_frames: int = 3
    team_similarity_threshold: float = 72.0
    team_debounce_frames: int = 2
    boundary_precision_sec: float = 0.05
    expected_starters: int = 11
    squad_similarity_threshold: float = 72.0
    write_split_clips: bool = True
    ocr: OCRConfig = field(default_factory=OCRConfig)
    player_detection: PlayerDetectionConfig = field(default_factory=PlayerDetectionConfig)

    def __post_init__(self) -> None:
        if self.sample_interval_sec <= 0 or self.stable_diff_threshold < 0:
            raise ValueError("Invalid frame sampling/stability configuration")
        if self.stable_min_frames < 2 or self.team_debounce_frames < 2:
            raise ValueError("Stable/debounce frame counts must be >= 2")
        if not 0 <= self.team_similarity_threshold <= 100 or not 0 <= self.squad_similarity_threshold <= 100:
            raise ValueError("Similarity thresholds must be in [0, 100]")
        if self.boundary_precision_sec <= 0 or self.expected_starters < 1:
            raise ValueError("Invalid boundary precision or expected starter count")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MappingConfig:
        regions = data.get("regions", data)
        pipeline = data.get("pipeline", {})
        ocr_data = data.get("ocr", {})
        detection_data = data.get("player_detection", {})

        def optional_rect(name: str) -> Rect | None:
            value = regions.get(name)
            return Rect.from_dict(value) if value else None

        return cls(
            team_name_bar=Rect.from_dict(regions["team_name_bar"]),
            lineup_region=Rect.from_dict(regions["lineup_region"]),
            substitutes_region=optional_rect("substitutes_region"),
            coach_region=optional_rect("coach_region"),
            exclude_regions=tuple(Rect.from_dict(value) for value in regions.get("exclude_regions", [])),
            sample_interval_sec=float(pipeline.get("sample_interval_sec", 0.3)),
            stable_diff_threshold=float(pipeline.get("stable_diff_threshold", 2.8)),
            stable_min_frames=int(pipeline.get("stable_min_frames", 3)),
            team_similarity_threshold=float(pipeline.get("team_similarity_threshold", 72)),
            team_debounce_frames=int(pipeline.get("team_debounce_frames", 2)),
            boundary_precision_sec=float(pipeline.get("boundary_precision_sec", 0.05)),
            expected_starters=int(pipeline.get("expected_starters", 11)),
            squad_similarity_threshold=float(pipeline.get("squad_similarity_threshold", 72)),
            write_split_clips=bool(pipeline.get("write_split_clips", True)),
            ocr=OCRConfig(**ocr_data),
            player_detection=PlayerDetectionConfig(**detection_data),
        )


def load_config(path: str | Path | None = None) -> MappingConfig:
    config_path = Path(path) if path else Path(__file__).with_name("layout_config.json")
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML config requires PyYAML: pip install PyYAML") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Layout config must contain an object at the top level")
    return MappingConfig.from_dict(data)
