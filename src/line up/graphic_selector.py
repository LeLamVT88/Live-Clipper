"""Compatibility facade for the starting-lineup mapping pipeline."""

from mapping.pipeline import MappingConfig as GraphicConfig
from mapping.pipeline import centered_candidates, extract_lineup_graphics

__all__ = ["GraphicConfig", "centered_candidates", "extract_lineup_graphics"]
