import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from legs import drive_block_maze


R0, C0, CELL = 21, 12, 3


def board_signature(frame):
    f = np.asarray(frame)
    rows = []
    for br in range(5):
        row = []
        for bc in range(13):
            block = f[R0 + br * CELL:R0 + (br + 1) * CELL,
                      C0 + bc * CELL:C0 + (bc + 1) * CELL]
            counts = Counter(map(int, block.ravel()))
            row.append("".join(f"{v}:{n}" for v, n in sorted(counts.items())))
        rows.append(" | ".join(row))
    return rows


def state_line(env):
    f = np.asarray(env.frame())
    unusual = {}
    for value in sorted(set(map(int, np.unique(f))) - {0, 2, 5, 6}):
        unusual[value] = [tuple(map(int, p)) for p in np.argwhere(f == value)]
    return {
        "level": int(env.levels_completed),
        "terminal": bool(env.terminal()),
        "unusual": unusual,
        "board": board_signature(f),
    }

def compact_state(env):
    f = np.asarray(env.frame())
    marked = []
    for br in range(5):
        for bc in range(13):
            block = f[R0 + br * CELL:R0 + (br + 1) * CELL,
                      C0 + bc * CELL:C0 + (bc + 1) * CELL]
            counts = Counter(map(int, block.ravel()))
            if any(v not in {0, 2, 5} for v in counts):
                marked.append(((br, bc), dict(sorted(counts.items()))))
    return (int(env.levels_completed), bool(env.terminal()), marked)


def probe(env):
    drive_block_maze(env)
    print("INITIAL", state_line(env))
    tests = [
        [1], [2], [3], [4],
        [1, 4, 4, 2, 4, 4],
        [1, 4, 4, 2, 4, 4, 1],
        [1, 4, 4, 2, 4, 4, 1, 4, 4, 1],
        [1, 4, 4, 2, 4, 1],
        [1, 4, 4, 2, 4, 4, 3],
        [1, 4, 4, 2, 4, 4, 2],
    ]
    for actions in tests:
        clone = env.clone()
        print("TEST", actions)
        for action in actions:
            if clone.terminal():
                break
            clone.step(action)
            print(" STEP", action, compact_state(clone))


result = A.run_program("tu93", probe)
print("RUN", result[0], "ERR", repr(result[2]))
