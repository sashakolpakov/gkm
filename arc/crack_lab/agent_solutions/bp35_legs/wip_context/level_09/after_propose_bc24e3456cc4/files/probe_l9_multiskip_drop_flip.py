"""Carry the last three-skip control down to the solid prize wall."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_multiskip_handoff import stage
from probe_l9_route_deletions import enter_level_9


DROP = (6, 27, 33)


def boxes(env, colors):
    return tuple(
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=colors, min_area=2)
        if blob.bbox[0] < 63
    )


def report(label, env):
    print(
        label,
        "terminal",
        bool(env.terminal()),
        "level",
        int(env.levels_completed) + 1,
        "avatar",
        boxes(env, (9,)),
        "controls",
        controls(env),
        "goals",
        boxes(env, (7,)),
        "grid",
        compact(env)["grid9"],
        flush=True,
    )


def dropped(root, depth):
    child = stage(root)
    for _ in range(depth):
        child.step(*DROP)
    return child


def run(root, depth, move):
    child = dropped(root, depth)
    visible = controls(child)
    report((depth, move, "PRE"), child)
    if not visible:
        return
    child.step(*visible[-1])
    report((depth, move, "FLIP"), child)
    for index in range(1, 9):
        child.step(move)
        report((depth, move, index), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    for depth in range(1, 6):
        for move in (3, 4):
            run(env, depth, move)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
