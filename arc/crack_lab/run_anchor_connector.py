"""Run the LLM-driven anchor connector on real games. Reports, per game, the
directed-probe effect summary, the LLM's ranked proposals, which the verifier
accepted/rejected, and the final anchor + source (llm-verified / algorithmic /
none)."""
from __future__ import annotations
import sys, time
from lab import make_env, avail
from anchor_connector import (AnchorConnector, directed_probe, scene_summary,
                              effect_summary)
from logical_grid import Grid
import numpy as np

GAMES = sys.argv[1:] or ["wa30", "g50t", "ls20", "tr87"]
USE_LLM = True


def run(game):
    print(f"\n================= {game} =================")
    acts, win = avail(game)
    acts = tuple(a for a in (acts or [1, 2, 3, 4, 5]) if 1 <= a <= 5)
    sequences, start = directed_probe(make_env(game), actions=acts)
    grid = Grid.infer(start)
    print(scene_summary(start, grid))
    print("directed-probe effects (per-component best mover):")
    print(effect_summary(sequences, grid))
    t0 = time.time()
    conn = AnchorConnector(use_llm=USE_LLM)
    res = conn.identify(make_env(game), actions=acts)
    print(f"\nLLM ranked ({time.time()-t0:.0f}s): {res.llm_ranked}")
    if res.rejected:
        print("rejected by verifier:", res.rejected)
    print(f"--> ANCHOR [{res.source}]:")
    print("   ", str(res.anchor).replace("\n", "\n    ") if res.anchor else "none")


for g in GAMES:
    try:
        run(g)
    except Exception as e:
        import traceback; print(f"{g} FAILED: {e}"); traceback.print_exc()
