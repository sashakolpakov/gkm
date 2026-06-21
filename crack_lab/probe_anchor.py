"""Probe the GENERAL PER-COMPONENT action-anchor across games (algorithmic
verifier only, no LLM). Uses the directed probe so mazes/gated games get a fair
shot. Tests 'separate the avatar, whatever it is' at component granularity."""
from __future__ import annotations
import sys
from lab import make_env, avail
from anchor_connector import directed_probe, effect_summary
from logical_grid import Grid
from cofibrant import identify_anchor

GAMES = sys.argv[1:] or ["wa30", "ls20", "g50t", "tr87"]

for game in GAMES:
    try:
        acts, _ = avail(game); acts = tuple(a for a in (acts or [1, 2, 3, 4, 5]) if 1 <= a <= 5)
        sequences, start = directed_probe(make_env(game), actions=acts)
    except Exception as e:
        print(f"\n=== {game} === could not run: {e}"); continue
    grid = Grid.infer(start)
    anchor = identify_anchor(sequences, grid)
    print(f"\n=== {game} ===  grid={grid}  actions={list(acts)}")
    print(effect_summary(sequences, grid))
    if anchor is None:
        print("  --> NO per-component anchor (honest null)")
    else:
        print("  -->", str(anchor).replace("\n", "\n  "))
