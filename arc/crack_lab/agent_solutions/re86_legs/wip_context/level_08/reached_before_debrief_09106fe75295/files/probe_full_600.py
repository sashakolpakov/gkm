import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from probe_l8_routes import step_many
from probe_l8_verify import FIRST_TRANSPORT, SECOND_TRANSPORT
from search_l7_segments import expand


LEVEL_4 = [
    (1, 10), (3, 8), (2, 8), (3, 5), (5, 1),
    (2, 12), (4, 7), (1, 9), (3, 2),
]
LEVEL_6 = [
    (1, 3), (4, 4), (1, 8), (4, 9), (2, 10), (5, 1),
    (1, 1), (3, 7), (2, 2), (4, 2), (1, 2), (3, 5),
    (2, 7), (1, 6),
]
LEVEL_7 = [
    (4, 10), (1, 7), (4, 1), (1, 4), (3, 6), (2, 1),
    (4, 6), (2, 7), (5, 1),
    (1, 3), (4, 3), (1, 2), (4, 5), (1, 6), (3, 10),
    (1, 4), (2, 4), (4, 11), (5, 1),
    (3, 2), (1, 8), (3, 3), (1, 3), (4, 3), (2, 3),
    (1, 3), (4, 6), (3, 6), (1, 4), (2, 3),
]


def optimized_prefix():
    with open("checkpoint.json") as handle:
        original = json.load(handle)["final_path"]
    return (
        original[:103]
        + expand(LEVEL_4)
        + original[171:234]
        + expand(LEVEL_6)
        + expand(LEVEL_7)
    )


def probe(env):
    prefix = optimized_prefix()
    level_8 = FIRST_TRANSPORT + SECOND_TRANSPORT
    print("lengths", len(prefix), len(level_8), len(prefix) + len(level_8))
    previous = env.levels_completed
    for index, action in enumerate(prefix + level_8, 1):
        env.step(action)
        if env.levels_completed != previous:
            print("reward", env.levels_completed, index)
            previous = env.levels_completed


if __name__ == "__main__":
    levels, path, error = A.run_program("re86", probe)
    valid = A.validate("re86", path, levels) if path else False
    print("result", levels, len(path), valid, error)
