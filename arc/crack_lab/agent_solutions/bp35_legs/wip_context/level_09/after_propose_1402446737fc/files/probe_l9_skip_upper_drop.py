"""Continue the fast one-skip column-four descent through the yellow region."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9
from probe_l9_skip_upper_stage2 import stage2


DROP = (6, 27, 33)


def pieces(env):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            env.frame(), colors=(7, 8, 9, 11, 12, 14, 15), min_area=2
        )
        if blob.bbox[0] < 63 and blob.color in (7, 8, 9, 14)
    )


def report(label, env):
    print(
        label,
        "terminal",
        bool(env.terminal()),
        "level",
        int(env.levels_completed) + 1,
        "controls",
        controls(env),
        "grid",
        compact(env)["grid9"],
        "pieces",
        pieces(env),
        flush=True,
    )


def dropped(root, count):
    child = stage2(root)
    for _ in range(count):
        child.step(*DROP)
    return child


def probe(env):
    enter_level_9(env)
    child = stage2(env)
    report(0, child)
    for depth in range(1, 14):
        child.step(*DROP)
        report(depth, child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break
    for depth in range(1, 7):
        root = dropped(env, depth)
        visible = controls(root)
        if not visible:
            continue
        for move in (3, 4):
            branch = root.clone()
            switch = visible[0]
            branch.step(*switch)
            branch.step(move)
            report(("FLIP_MOVE", depth, switch, move), branch)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
