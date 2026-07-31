"""Print exact logical-cell shapes for fixed ports on selected sp80 levels."""
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import perception as P
from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)


def logical_shape(frame, bbox, cell):
    f = P.arr(frame)
    r0, c0, r1, c1 = bbox
    rows = []
    for r in range(r0, r1 + 1, cell):
        row = []
        for c in range(c0, c1 + 1, cell):
            vals, counts = np.unique(
                f[r:min(r + cell, r1 + 1), c:min(c + cell, c1 + 1)],
                return_counts=True,
            )
            row.append(int(vals[counts.argmax()]))
        rows.append(tuple(row))
    return tuple(rows)


def ports(env, cell):
    return tuple(
        (o["bbox"], logical_shape(env.frame(), o["bbox"], cell))
        for o in P.object_candidates(env.frame(), min_area=4)
        if o["color"] in (4, 6, 11)
    )


def probe(env):
    print("PORT_SHAPES", 1, ports(env, 4))
    play_level_1(env)
    print("PORT_SHAPES", 2, ports(env, 4))
    for player in (play_level_2, play_level_3, play_level_4, play_level_5):
        player(env)
    print("PORT_SHAPES", 6, ports(env, 3))


if __name__ == "__main__":
    levels, path, err = A.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)
