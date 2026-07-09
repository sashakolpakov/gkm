"""Pytest path setup for domain subdirectories.

The repository is organised into domain directories (``foraging/``,
``bongard/``, ``transduction/``, ``arc/``, ``cone/``). Tests and experiment
runners import their sibling modules by bare top-level name (e.g. ``import
evo_game``, ``import cone_foraging``, ``import arc_agi3_adapter``).
Putting each domain directory on ``sys.path`` keeps those imports resolving
under pytest without any packaging boilerplate — including cross-domain
imports (the arc modules build on the cone core).
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _domain in ("foraging", "bongard", "transduction", "arc", "cone"):
    _path = _ROOT / _domain
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
