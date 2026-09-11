"""Select lineup graphics and read starters from an existing clip."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mapping.graphic_cli import main

if __name__ == "__main__":
    main()
