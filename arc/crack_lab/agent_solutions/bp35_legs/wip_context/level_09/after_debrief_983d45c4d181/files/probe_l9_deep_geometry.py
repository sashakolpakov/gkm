"""Describe raw wall runs around the visible deep prize chamber."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_multi_skip_bottom import root_for
from probe_l9_route_deletions import enter_level_9


SYMBOLS = {3: "#", 5: "#", 7: "G", 9: "A", 10: " ", 11: "a", 15: "*"}


def runs(row):
    out = []
    start = 0
    current = SYMBOLS.get(int(row[0]), "?")
    for x in range(1, 63):
        symbol = SYMBOLS.get(int(row[x]), "?")
        if symbol != current:
            out.append((start, x - 1, current))
            start = x
            current = symbol
    out.append((start, 62, current))
    return tuple(out)


def probe(env):
    enter_level_9(env)
    child = root_for(env, 6)
    print(
        "WALL_COMPONENTS",
        tuple(
            (blob.color, blob.bbox, blob.area)
            for blob in connected_components(
                child.frame(), colors=(3, 5), min_area=2
            )
            if blob.bbox[0] < 63
        ),
    )
    frame = child.frame()
    for y in range(24, 51):
        print("ROW", y, runs(frame[y]))


if __name__ == "__main__":
    arena.run_program("bp35", probe)
