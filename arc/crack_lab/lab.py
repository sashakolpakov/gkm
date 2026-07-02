"""Shared harness for the game-cracking lab (scratch; gitignored).

Imports the gkm colimit machinery and points LocalArcEnv at the repo's
downloaded games. Everything here just RUNS the construction against local
games and prints what it finds — no LLM prose substituting for the search.
"""
from __future__ import annotations
import logging, os, sys
from pathlib import Path

GKM = Path(__file__).resolve().parents[2]
if str(GKM) not in sys.path:
    sys.path.insert(0, str(GKM))
ENVDIR = str(GKM / "environment_files")

# load key (only needed to DOWNLOAD a game once; offline play needs none)
_env = GKM / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

logging.disable(logging.INFO)

import arc_agi3_adapter as arc  # noqa: E402

DOWNLOADED = sorted({p.parent.parent.name for p in Path(ENVDIR).glob("*/*/metadata.json")}) if Path(ENVDIR).exists() else []


def make_env(game: str, mode: str = "offline"):
    return lambda: arc.LocalArcEnv(game, operation_mode=mode, environments_dir=ENVDIR)


def avail(game: str):
    e = arc.LocalArcEnv(game, "offline", environments_dir=ENVDIR)
    s = e.reset()
    return e.available_actions, s.win_levels
