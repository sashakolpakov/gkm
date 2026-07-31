import json, sys, traceback
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
from perception import arr, connected_components, UP, DOWN, LEFT, RIGHT, USE
from collections import Counter

with open("checkpoint.json") as f:
    PATH = json.load(f)["final_path"]

RING = 4
BORDER = 2


def rings(f):
    out = []
    for b in connected_components(f, colors=(RING,), min_area=8):
        if b.size == (3, 3) and b.area == 8:
            r0, c0 = b.bbox[0], b.bbox[1]
            out.append(((r0 + 1, c0 + 1), int(f[r0 + 1, c0 + 1])))
    return sorted(out)


def blackpx(f):
    bp = list(zip(*((f == 0).nonzero())))
    return tuple(int(v) for v in bp[0]) if len(bp) == 1 else None


def probe_shape(node, bg):
    """Return (center, color, sorted offsets) of currently selected shape."""
    before = arr(node.frame()).copy()
    center = blackpx(before)
    votes = []
    offs = set()
    frames = []
    for a in (UP, DOWN, LEFT, RIGHT):
        m = node.clone(); m.step(a)
        af = arr(m.frame())
        frames.append(af)
        votes.extend(int(before[r, c]) for r, c in zip(*((before != af).nonzero()))
                     if int(af[r, c]) == bg and int(before[r, c]) not in (0, bg, BORDER))
    if not votes:
        return center, None, set()
    col = Counter(votes).most_common(1)[0][0]
    for af in frames:
        offs.update((int(r) - center[0], int(c) - center[1])
                    for r, c in zip(*((before != af).nonzero()))
                    if int(before[r, c]) == col and int(af[r, c]) == bg)
    return center, col, offs


def classify(offs):
    if not offs:
        return None
    ax = sum(r == 0 or c == 0 for r, c in offs)
    xs = sum(abs(r) == abs(c) for r, c in offs)
    rad = max(max(abs(r), abs(c)) for r, c in offs)
    dia = max(abs(r) + abs(c) for r, c in offs)
    rspan = max(r for r, _ in offs) - min(r for r, _ in offs)
    cspan = max(c for _, c in offs) - min(c for _, c in offs)
    kind = "?"
    if ax == len(offs):
        kind = "plus" if (rspan and cspan) else ("vertical" if rspan else "horizontal")
    elif xs == len(offs):
        kind = "x"
    elif all(abs(r) + abs(c) == dia for r, c in offs):
        kind = "diamond"
    return dict(kind=kind, rad=rad, dia=dia, n=len(offs),
                rmin=min(r for r, _ in offs), rmax=max(r for r, _ in offs),
                cmin=min(c for _, c in offs), cmax=max(c for _, c in offs),
                ax=ax, xs=xs)


def solve(env):
    for a in PATH:
        env.step(a)
    f = arr(env.frame())
    bg = Counter(int(v) for v in f.flat).most_common(1)[0][0]
    print("bg =", bg)
    print("rings (center, color):")
    for p, c in rings(f):
        print("   ", p, c)

    # stations
    print("stations:")
    for b in connected_components(f, colors=(BORDER,), min_area=20):
        if b.size == (6, 6) and b.area == 20:
            r0, c0, r1, c1 = b.bbox
            inner = f[r0+1:r1, c0+1:c1]
            cs = {int(v) for v in inner.flat if int(v) not in (bg, BORDER)}
            print("   ", b.bbox, cs)

    # cycle selection
    node = env.clone()
    first = None
    for i in range(12):
        center, col, offs = probe_shape(node, bg)
        if first is None:
            first = center
        elif center == first:
            print(f"  cycle closed after {i} shapes")
            break
        print(f"  shape {i}: center={center} color={col} {classify(offs)}")
        node.step(USE)
    raise SystemExit


try:
    A.run_program('re86', solve)
except SystemExit:
    pass
except Exception:
    traceback.print_exc()
