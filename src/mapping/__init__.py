"""Starting-lineup extraction from short football graphic clips."""

from mapping.config import MappingConfig, load_config
from mapping.pipeline import extract_lineup

__all__ = ["MappingConfig", "extract_lineup", "load_config"]
