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


def go(node, dr, dc):
    for _ in range(abs(dr) // 3):
        node.step(DOWN if dr > 0 else UP)
    for _ in range(abs(dc) // 3):
        node.step(RIGHT if dc > 0 else LEFT)


def state(node, tag):
    f = arr(node.frame())
    cc = Counter(int(v) for v in f.flat)
    print(f"[{tag}] center={blackpx(f)} lvl={node.levels_completed} c4={cc.get(4,0)}")
    print("      rings:", rings(f))


def run(base, paint, tag):
    n = base.clone()
    if paint:
        go(n, 54 - 33, 57 - 54)   # 8-station
        go(n, 33 - 54, 54 - 57)   # back
    state(n, tag + " @(33,54)")
    go(n, -12, 0)                 # move plus up to (21,54), unveiling
    state(n, tag + " moved away")


def solve(env):
    for a in PATH:
        env.step(a)
    base = env.clone()
    for _ in range(2):
        base.step(USE)
    run(base, False, "CONTROL color12")
    run(base, True, "PAINTED color8")
    raise SystemExit


try:
    A.run_program('re86', solve)
except SystemExit:
    pass
except Exception:
    traceback.print_exc()
