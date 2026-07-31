"""Test the small set of meaningful four-action routes at the lane-six frontier."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact
from probe_l9_final_bfs import frontier, step
from probe_l9_route_deletions import enter_level_9


YELLOW = (6, 27, 17)
BELOW = (6, 39, 33)
LEFT_CATCH = (6, 33, 27)
RIGHT_CATCH = (6, 51, 27)


def objects(env):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            env.frame(), colors=(7, 8, 9, 11, 12, 14, 15), min_area=2
        )
        if blob.bbox[0] < 63
        and (blob.color != 15 or blob.area == 21)
    )


def probe(env):
    enter_level_9(env)
    root = frontier(env)
    print("ROOT", compact(root), "objects", objects(root), flush=True)
    variants = {
        "RRRR": (4, 4, 4, 4),
        "YRRR": (YELLOW, 4, 4, 4),
        "RYRR": (4, YELLOW, 4, 4),
        "BRRR": (BELOW, 4, 4, 4),
        "YBRR": (YELLOW, BELOW, 4, 4),
        "RCRR": (4, RIGHT_CATCH, 4, 4),
        "YLLL": (YELLOW, 3, 3, 3),
        "CLLL": (LEFT_CATCH, 3, 3, 3),
    }
    for name, actions in variants.items():
        child = root.clone()
        for index, action in enumerate(actions, 1):
            step(child, action)
            if int(child.levels_completed) >= 9:
                print("WIN", name, index, action, compact(child), flush=True)
                return
            if child.terminal():
                break
        print(
            "END",
            name,
            "levels",
            int(child.levels_completed),
            "terminal",
            bool(child.terminal()),
            "state",
            compact(child),
            "objects",
            objects(child),
            flush=True,
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
