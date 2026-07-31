"""Map local exits after flipping early from lower-maze column seven."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_pre_deep_flip_lanes import before_flip
from probe_l9_route_deletions import enter_level_9


def objects(env):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            env.frame(), colors=(7, 8, 9, 11, 12, 14, 15), min_area=2
        )
        if blob.bbox[0] < 63 and (blob.color != 15 or blob.area == 21)
    )


def root_lane7(root):
    child = before_flip(root)
    child.step(3)
    child.step(3)
    child.step(*controls(child)[0])
    return child


def report(label, env):
    print(
        label,
        "terminal",
        bool(env.terminal()),
        "level",
        int(env.levels_completed) + 1,
        "controls",
        controls(env),
        "state",
        compact(env),
        "objects",
        objects(env),
    )


def run(root, name, prefix, actions):
    child = root_lane7(root)
    for action in prefix:
        child.step(*action) if isinstance(action, tuple) else child.step(action)
    report((name, "START"), child)
    for action in actions:
        branch = child.clone()
        branch.step(*action) if isinstance(action, tuple) else branch.step(action)
        report((name, "ACTION", action), branch)


def probe(env):
    enter_level_9(env)
    run(
        env,
        "C7",
        (),
        (
            3,
            4,
            7,
            (6, 45, 33),
            (6, 45, 39),
            (6, 45, 45),
            (6, 39, 33),
            (6, 39, 39),
            (6, 39, 45),
        ),
    )
    run(
        env,
        "C4",
        (3, 3, 3),
        (
            3,
            4,
            7,
            (6, 21, 39),
            (6, 27, 33),
            (6, 27, 39),
            (6, 27, 45),
            (6, 21, 33),
            (6, 21, 45),
        ),
    )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
