import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import action_deltas, color_counts, connected_components


def reach_level_6(env):
    for level in range(1, 6):
        getattr(players, f"play_level_{level}")(env)
        assert env.levels_completed == level


def macro(frame, cell=4):
    a = np.asarray(frame)
    rows = []
    for r in range(0, a.shape[0], cell):
        row = []
        for c in range(0, a.shape[1], cell):
            vals, counts = np.unique(a[r:r + cell, c:c + cell], return_counts=True)
            row.append(f"{int(vals[counts.argmax()]):X}")
        rows.append("".join(row))
    return rows


def probe(env):
    reach_level_6(env)
    print("LEVEL", env.levels_completed + 1, "ACTIONS", env.actions)
    print("COUNTS", color_counts(env.frame()))
    comps = connected_components(env.frame(), min_area=4)
    print("BLOBS", [(b.color, b.bbox, b.area) for b in comps])
    print("MACRO")
    print("\n".join(macro(env.frame())))
    deltas = action_deltas(env, actions=env.actions)
    print("DELTAS", {a: (d["count"], d["bbox"]) for a, d in deltas.items()})
    first = [2] + [4] * 6 + [3] * 4 + [1] * 2 + [4] + [1] * 2 + [3] * 3
    second = [2] * 5 + [4] * 6 + [3] * 4 + [1] * 5 + [4] + [3] * 3
    third = [2] * 6 + [4] * 6 + [3] * 4 + [1] * 6 + [4] + [3] * 3
    tests = [
        [(6, 32, 5)],
        [(6, 32, 5), 1],
        [(6, 32, 5), 2],
        [(6, 32, 5), 3],
        [(6, 32, 5), 4],
        [(6, 8, 29), 1],
        [(6, 32, 35), 4],
    ]
    for path in tests:
        node = env.clone()
        for action in path:
            if isinstance(action, tuple):
                node.step(*action)
            else:
                node.step(action)
        pieces = [
            (b.color, b.bbox, b.area)
            for b in connected_components(node.frame(), colors=(0, 1, 6, 8, 9, 15), min_area=4)
        ]
        print("TEST", path, "L", node.levels_completed, "P", pieces)


A.run_program("sk48", probe)
