"""Pre-stage the column-nine upper-chamber landing from early-flip column eight."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_lane7_early_flip_local import objects
from probe_l9_lane8_early_flip_exit import root_lane8
from probe_l9_route_deletions import enter_level_9


def report(label, env):
    avatars = tuple(
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(9,), min_area=3)
        if blob.bbox[0] < 63
    )
    print(
        label,
        "terminal",
        bool(env.terminal()),
        "level",
        int(env.levels_completed) + 1,
        "avatars",
        avatars,
        "controls",
        controls(env),
        "state",
        compact(env),
        "objects",
        objects(env),
    )


def probe(env):
    enter_level_9(env)
    sequences = {
        "DIRECT": (),
        "ROW7": ((6, 57, 45),),
        "UP1": ((6, 57, 45), (6, 57, 39)),
        "UP2": ((6, 57, 45), (6, 57, 39), (6, 57, 33)),
        "C8_THEN_C9": ((6, 51, 45), (6, 57, 45)),
    }
    for name, prefix in sequences.items():
        child = root_lane8(env)
        for action in prefix:
            child.step(*action)
        report((name, "STAGED"), child)
        child.step(4)
        report((name, "ENTER"), child)
        if child.terminal():
            continue
        for action in ((6, 57, 33), (6, 57, 39), (6, 57, 45)):
            branch = child.clone()
            branch.step(*action)
            report((name, "VERTICAL", action), branch)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
