from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from mapping.config import parse_config
from mapping.frames import LineupOCRError
from mapping.pipeline import LineupWorkflow
from mapping.resolver import LineupResolutionError
from mapping.selector import LineupFrameSelectionError


def main() -> int:
    try:
        return LineupWorkflow(parse_config()).run()
    except (
        LineupFrameSelectionError,
        LineupOCRError,
        LineupResolutionError,
        OSError,
        pd.errors.ParserError,
        ValueError,
    ) as exc:
        print(f"Lineup pipeline failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
