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


def blackpx(f):
    bp = list(zip(*((f == 0).nonzero())))
    return tuple(int(v) for v in bp[0]) if len(bp) == 1 else None


def solve(env):
    for a in PATH:
        env.step(a)
    base = env.clone()
    # select plus: USE twice (order X, diamond, plus)
    for _ in range(2):
        base.step(USE)
    print("selected center:", blackpx(arr(base.frame())))

    for label, moves in [("left4", [LEFT] * 4), ("up4", [UP] * 4),
                         ("left4+up4", [LEFT] * 4 + [UP] * 4)]:
        n = base.clone()
        for a in moves:
            n.step(a)
        f = arr(n.frame())
        print(f"--- after {label}: center={blackpx(f)} lvl={n.levels_completed}")
        for p, c in rings(f):
            print("   ", p, c)
    raise SystemExit


try:
    A.run_program('re86', solve)
except SystemExit:
    pass
except Exception:
    traceback.print_exc()
