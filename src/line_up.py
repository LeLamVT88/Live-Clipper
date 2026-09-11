"""Expose the ``line up`` directory as the importable ``line_up`` package."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

_package_dir = Path(__file__).resolve().parent / "line up"
_spec = spec_from_file_location(
    __name__, _package_dir / "__init__.py",
    submodule_search_locations=[str(_package_dir)],
)
_package = module_from_spec(_spec)
sys.modules[__name__] = _package
_spec.loader.exec_module(_package)
