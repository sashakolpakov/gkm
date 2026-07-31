import json, sys, traceback
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from perception import arr, connected_components, UP, DOWN, LEFT, RIGHT, USE
from collections import Counter

with open("checkpoint.json") as f:
    PATH = json.load(f)["final_path"]


def rings(f):
    out = []
    for b in connected_components(f, colors=(4,), min_area=8):
        if b.size == (3, 3) and b.area == 8:
            r0, c0 = b.bbox[0], b.bbox[1]
            out.append(((r0 + 1, c0 + 1), int(f[r0 + 1, c0 + 1])))
    return sorted(out)


def ring4cells(f):
    """count color-4 cells that look like ring fragments (any 3x3 ring remnants)"""
    return int((f == 4).sum())


def blackpx(f):
    bp = list(zip(*((f == 0).nonzero())))
    return tuple(int(v) for v in bp[0]) if len(bp) == 1 else None


def go(node, dr, dc):
    v, h = (DOWN, RIGHT)
    for _ in range(abs(dr) // 3):
        node.step(DOWN if dr > 0 else UP)
    for _ in range(abs(dc) // 3):
        node.step(RIGHT if dc > 0 else LEFT)


def state(node, tag):
    f = arr(node.frame())
    cc = Counter(int(v) for v in f.flat)
    print(f"[{tag}] center={blackpx(f)} lvl={node.levels_completed} "
          f"c4={cc.get(4,0)} colors={ {k: v for k, v in sorted(cc.items()) if k not in (5,)} }")
    print("      rings:", rings(f))


def solve(env):
    for a in PATH:
        env.step(a)
    base = env.clone()
    for _ in range(2):
        base.step(USE)          # select the plus
    state(base, "plus selected @(33,54)")

    # walk to the color-8 station interior: (54,57)
    n = base.clone()
    go(n, 54 - 33, 57 - 54)
    state(n, "at 8-station (54,57)")

    # back to (33,54)
    go(n, 33 - 54, 54 - 57)
    state(n, "repainted, back @(33,54)")
    raise SystemExit


try:
    A.run_program('re86', solve)
except SystemExit:
    pass
except Exception:
    traceback.print_exc()
