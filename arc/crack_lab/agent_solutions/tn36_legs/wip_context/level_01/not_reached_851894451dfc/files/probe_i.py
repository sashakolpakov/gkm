"""Does the board effect follow the (x,y) clicked, or the clone-step index?

MODE=fwd    : sweep in the same order as probe_c
MODE=rev    : sweep in reverse order
MODE=single : one single clone-step in a fresh env
"""
import sys
import numpy as np
sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

MODE = sys.argv[1]


def cls(a, b):
    """Classify the non-budget delta compactly."""
    a, b = np.asarray(a), np.asarray(b)
    m = (a != b); m[1, :] = False
    ys, xs = np.where(m)
    if not len(ys):
        return "-"
    n = len(ys)
    return f"{n}@({ys.min()},{xs.min()})-({ys.max()},{xs.max()})"


def run(env):
    base = np.asarray(env.frame()).copy()
    coords = list(range(0, 64, 4))
    pts = [(a, b) for a in coords for b in coords]
    if MODE == "rev":
        pts = pts[::-1]
    if MODE == "single":
        pts = [(36, 20)]
    for i, (x, y) in enumerate(pts):
        c = env.clone()
        c.step(6, x, y)
        k = cls(base, c.frame())
        if MODE == "single" or k != "-":
            print(f"i={i:3d} x={x:2d} y={y:2d} {k}")


A.run_program("tn36", run)
