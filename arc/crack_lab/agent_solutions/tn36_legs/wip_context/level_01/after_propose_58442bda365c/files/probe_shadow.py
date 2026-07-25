"""Explore on ONE long-lived clone ("shadow env"), which reflects the live engine.

usage: python probe_shadow.py "x,y x,y ..." [--ascii] [--root]
"""
import sys
import numpy as np
sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
from perception import connected_components

SEQ = []
for tok in (sys.argv[1].split() if len(sys.argv) > 1 and sys.argv[1] else []):
    x, y = tok.split(",")
    SEQ.append((int(x), int(y)))
ASCII = "--ascii" in sys.argv
ROOT = "--root" in sys.argv
LEG = {0: '.', 1: '1', 3: '3', 4: '4', 5: ' ', 9: '9', 11: 'C'}


def budget(f):
    return int((f[1, 1:62] == 9).sum())


def glyphs(f):
    return "".join('1' if int(f[44, c]) == 1 else '0' for c in (21, 26, 31, 36, 41))


def cyan(f):
    return [b.bbox for b in connected_components(f, colors=[11], min_area=1)]


def dump(f, r0=6, r1=62, c0=12, c1=52):
    print("        " + "".join(str(c % 10) for c in range(c0, c1)))
    for r in range(r0, r1):
        print(f"     {r:3d} " + "".join(LEG.get(int(v), '?') for v in f[r, c0:c1]))


def run(env):
    e = env if ROOT else env.clone()
    prev = np.asarray(e.frame()).copy()
    print(f"init B={budget(prev)} G={glyphs(prev)} cyan={cyan(prev)} lvl={e.levels_completed}")
    if ASCII:
        dump(prev)
    for i, (x, y) in enumerate(SEQ):
        e.step(6, x, y)
        f = np.asarray(e.frame()).copy()
        m = (f != prev); m[1, :] = False
        n = int(m.sum())
        print(f"{i:3d}:({x:2d},{y:2d}) n={n:3d} B={budget(f):2d} G={glyphs(f)} "
              f"lvl={e.levels_completed} term={e.terminal()} cyan={cyan(f)}")
        if ASCII and n:
            dump(f)
        prev = f
        if e.terminal():
            print("TERMINAL")
            break


A.run_program("tn36", run)
