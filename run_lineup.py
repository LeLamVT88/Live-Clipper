"""Run the lineup detection pipeline from src."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from line_up.cli import main

if __name__ == "__main__":
    main()
