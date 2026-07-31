import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import arr
from probe48 import DOWN, LEFT, PATH, RIGHT, UP, USE, bare_frame, covered, descriptor


MOVES = ((-6, 0), (0, -21), (9, 0), (0, 9), (-9, 0), (0, -15), (24, 0), (-18, 0))
ACTION = {(-3, 0): UP, (3, 0): DOWN, (0, -3): LEFT, (0, 3): RIGHT}
TARGETS = ((9, 9), (15, 3), (15, 36), (27, 9))


def transform(point, variant):
    row, col = point
    if variant & 1:
        row = -row
    if variant & 2:
        col = -col
    if variant & 4:
        row, col = col, row
    return row, col


def actions(variant):
    out = []
    for delta in MOVES:
        dr, dc = transform(delta, variant)
        action = ACTION[(0 if dr == 0 else (3 if dr > 0 else -3), 0 if dc == 0 else (3 if dc > 0 else -3))]
        out.extend([action] * ((abs(dr) + abs(dc)) // 3))
    return out


def run(env):
    for action in PATH:
        env.step(action)
    root = env.clone()
    root.step(USE)
    root.step(USE)
    bare = bare_frame(root)
    for variant in range(8):
        node = root.clone()
        for action in actions(variant):
            node.step(action)
        total_row = sum(transform(delta, variant)[0] for delta in MOVES)
        total_col = sum(transform(delta, variant)[1] for delta in MOVES)
        center = (54 + total_row, 24 + total_col)
        desc = descriptor(arr(node.frame()), bare, "large-cross", center)
        print(variant, desc[:3] if desc else None, covered(desc, "large-cross", TARGETS))


A.run_program("re86", run)
