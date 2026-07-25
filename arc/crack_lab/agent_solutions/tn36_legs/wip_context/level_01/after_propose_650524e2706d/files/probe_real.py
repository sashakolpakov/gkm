"""Apply a click sequence to the ROOT env (no clones) and report state changes.

usage: python probe_real.py "x,y x,y ..."   [--ascii]
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
LEGEND = {0: '.', 1: '1', 3: '3', 4: '4', 5: ' ', 9: '9', 11: 'C'}


def budget(f):
    return int((f[1, 1:62] == 9).sum())


def glyphs(f):
    return "".join(str(int(f[44, c])) for c in (21, 26, 31, 36, 41))


def cyan(f):
    return [(b.bbox, b.area) for b in connected_components(f, colors=[11], min_area=1)]


def dump(f, r0=8, r1=51, c0=12, c1=52):
    print("      " + "".join(str(c % 10) for c in range(c0, c1)))
    for r in range(r0, r1):
        print(f"   {r:3d} " + "".join(LEGEND.get(int(v), '?') for v in f[r, c0:c1]))


def brief(tag, f, env):
    print(f"{tag:>18s} B={budget(f):2d} G={glyphs(f)} lvl={env.levels_completed} "
          f"term={env.terminal()} cyan={cyan(f)}")


def run(env):
    prev = np.asarray(env.frame()).copy()
    brief("init", prev, env)
    if ASCII:
        dump(prev)
    for i, (x, y) in enumerate(SEQ):
        env.step(6, x, y)
        f = np.asarray(env.frame()).copy()
        m = (f != prev); m[1, :] = False
        n = int(m.sum())
        brief(f"{i}:({x},{y}) n={n}", f, env)
        if n and ASCII:
            dump(f)
        prev = f
        if env.terminal():
            print("TERMINAL")
            break


A.run_program("tn36", run)
