import json, sys, traceback
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from perception import arr, connected_components, UP, DOWN, LEFT, RIGHT, USE
from collections import Counter

with open("checkpoint.json") as f:
    PATH = json.load(f)["final_path"]

STATION_PT = {11: (6, 6), 10: (6, 57), 14: (30, 6), 9: (54, 6), 8: (54, 57)}
BG = 5


def blackpx(f):
    bp = list(zip(*((f == 0).nonzero())))
    return tuple(int(v) for v in bp[0]) if len(bp) == 1 else None


def go(node, dr, dc):
    for _ in range(abs(dr) // 3):
        node.step(DOWN if dr > 0 else UP)
    for _ in range(abs(dc) // 3):
        node.step(RIGHT if dc > 0 else LEFT)


def shape_info(node):
    """(center, color, offsets, has_center_cell) of selected shape."""
    before = arr(node.frame()).copy()
    center = blackpx(before)
    votes, offs, frames = [], set(), []
    for a in (UP, DOWN, LEFT, RIGHT):
        m = node.clone(); m.step(a)
        af = arr(m.frame())
        frames.append(af)
        votes.extend(int(before[r, c]) for r, c in zip(*((before != af).nonzero()))
                     if int(af[r, c]) == BG and int(before[r, c]) not in (0, BG, 2))
    if not votes:
        return center, None, set(), None
    col = Counter(votes).most_common(1)[0][0]
    for af in frames:
        offs.update((int(r) - center[0], int(c) - center[1])
                    for r, c in zip(*((before != af).nonzero()))
                    if int(before[r, c]) == col and int(af[r, c]) == BG)
    d = node.clone(); d.step(USE)
    has_center = int(arr(d.frame())[center]) == col
    return center, col, offs, has_center


def describe(offs):
    if not offs:
        return "empty"
    ax = sum(r == 0 or c == 0 for r, c in offs)
    xs = sum(abs(r) == abs(c) for r, c in offs)
    dia = max(abs(r) + abs(c) for r, c in offs)
    rad = max(max(abs(r), abs(c)) for r, c in offs)
    kind = ("plus" if ax == len(offs) else "x" if xs == len(offs)
            else "diamond" if all(abs(r) + abs(c) == dia for r, c in offs) else "?")
    return f"{kind} rad={rad} dia={dia} n={len(offs)}"


def cycle(node, tag):
    n = node.clone()
    first, out = None, []
    for _ in range(8):
        c, col, offs, hc = shape_info(n)
        if first is None:
            first = c
        elif c == first:
            break
        out.append(f"{col}@{c} {describe(offs)} ctr={hc}")
        n.step(USE)
    print(f"  [{tag}] {len(out)} shapes: " + " | ".join(out))


def solve(env):
    for a in PATH:
        env.step(a)
    base = env.clone()
    cycle(base, "INITIAL")
    for s in range(3):
        sel = base.clone()
        for _ in range(s):
            sel.step(USE)
        c0 = blackpx(arr(sel.frame()))
        for k, pt in STATION_PT.items():
            n = sel.clone()
            go(n, pt[0] - c0[0], pt[1] - c0[1])
            cycle(n, f"shape{s}@{c0} painted {k}")
    raise SystemExit


try:
    A.run_program('re86', solve)
except SystemExit:
    pass
except Exception:
    traceback.print_exc()
