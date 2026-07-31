import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr


def runs(values, colors=(2, 15)):
    out = []
    start = None
    current = None
    for index, value in enumerate(values):
        value = int(value)
        kind = value if value in colors else None
        if kind != current:
            if current is not None:
                out.append((current, start, index - 1))
            start = index if kind is not None else None
            current = kind
    if current is not None:
        out.append((current, start, len(values) - 1))
    return tuple(out)


def probe(env):
    for level in range(1, 7):
        getattr(players, f"play_level_{level}")(env)
    frame = arr(env.frame())[:63, :63]
    for row in range(63):
        found = runs(frame[row, :])
        if found:
            print("row", row, found)
    for col in range(63):
        found = runs(frame[:, col])
        if found:
            print("col", col, found)


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
