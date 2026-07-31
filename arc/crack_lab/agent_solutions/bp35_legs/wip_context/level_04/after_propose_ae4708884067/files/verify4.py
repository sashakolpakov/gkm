"""Independent replay verification of the discovered level-4 path."""
import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from legs import band_grid


PREFIX = json.load(open("level4_prefix.json"))
PATH = [
    (4,), (4,), (6, 33, 3), (3,), (3,), (3,), (6, 21, 33),
    (4,), (4,), (6, 21, 39), (4,), (4,), (6, 33, 57),
    (6, 45, 33), (6, 45, 33), (3,), (3,), (3,), (6, 27, 39),
]


def probe(env):
    for action in PREFIX:
        env.step(*(action if isinstance(action, list) else [action]))
    assert env.levels_completed == 3
    for n, action in enumerate(PATH, 1):
        env.step(*action)
        grid = "/".join("".join(row) for row in band_grid(env.frame()))
        print(n, action, env.levels_completed, env.terminal(), grid)
        if env.terminal() or env.levels_completed > 3:
            break


print(A.run_program("bp35", probe))
