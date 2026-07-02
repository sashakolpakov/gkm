"""Compatibility entry point for the foraging ecology experiment."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_foraging_ecology import main  # noqa: E402


if __name__ == "__main__":
    main()
