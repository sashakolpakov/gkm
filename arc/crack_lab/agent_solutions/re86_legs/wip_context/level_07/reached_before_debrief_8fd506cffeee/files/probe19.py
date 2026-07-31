import json, sys, traceback
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from perception import arr, UP, DOWN, LEFT, RIGHT, USE
from collections import Counter

with open("checkpoint.json") as f:
    PATH = json.load(f)["final_path"]


def blackpx(f):
    bp = list(zip(*((f == 0).nonzero())))
    return tuple(int(v) for v in bp[0]) if len(bp) == 1 else None


def solve(env):
    for a in PATH:
        env.step(a)
    n = env.clone()
    moves = []
    # X: (42,24) -> paint 9 @(54,6) -> (15,30)
    moves += [DOWN] * 4 + [LEFT] * 6
    moves += [RIGHT] * 8 + [UP] * 13
    moves += [USE]
    # DIA: (18,30) -> paint 8 @(54,57) -> (36,51)
    moves += [DOWN] * 12 + [RIGHT] * 9
    moves += [UP] * 6 + [LEFT] * 2
    moves += [USE]
    # PLUS: (33,54) -> paint 9 @(54,6) -> (51,33)
    moves += [LEFT] * 16 + [DOWN] * 7
    moves += [UP] * 1 + [RIGHT] * 9

    for i, a in enumerate(moves):
        before = n.levels_completed
        n.step(a)
        if n.levels_completed != before:
            print(f"*** LEVEL UP at move {i} -> {n.levels_completed}")
    f = arr(n.frame())
    print("moves:", len(moves), "center:", blackpx(f), "lvl:", n.levels_completed,
          "terminal:", n.terminal())
    print("colors:", {k: v for k, v in sorted(Counter(int(v) for v in f.flat).items())})
    raise SystemExit


try:
    A.run_program('re86', solve)
except SystemExit:
    pass
except Exception:
    traceback.print_exc()
