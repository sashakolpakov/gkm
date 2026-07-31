"""Use the seven-drop compression to carry the boosted switch one shaft deeper."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9, replay, route
from probe_l9_twelve_fast_frontier import SKIPS


FAST_SKIPS = SKIPS | set(range(11, 14)) | set(range(103, 110))


def boxes(env, color):
    return tuple(
        blob.bbox
        for blob in connected_components(env.frame(), colors=(color,), min_area=2)
        if blob.bbox[0] < 63
    )


def report(label, env):
    print(
        label,
        "terminal",
        bool(env.terminal()),
        "levels",
        int(env.levels_completed),
        "controls",
        controls(env),
        "goals",
        boxes(env, 7),
        "avatar",
        boxes(env, 9),
        "grid",
        compact(env)["grid9"],
        flush=True,
    )


def pre_final(root):
    child = replay(root, route(), skips=FAST_SKIPS)
    for action in (
        (6, 21, 39),
        4,
        (6, 27, 39),
        4,
        (6, 27, 33),
        (6, 27, 33),
        (6, 27, 33),
    ):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
    child.step(*controls(child)[0])
    child.step(*controls(child)[-1])
    for action in ((6, 21, 39), 3, (6, 15, 39), 3):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
    child.step(*controls(child)[0])
    child.step(4)
    child.step(4)
    child.step(6, 27, 33)
    report("PRE_FINAL", child)
    return child


def final_flip(root):
    child = pre_final(root)
    child.step(*controls(child)[0])
    return child


def run(root, name, actions):
    child = final_flip(root)
    report((name, 0), child)
    for index, action in enumerate(actions, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        report((name, index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    variants = {
        "LEFT": (3,) * 12,
        "RIGHT": (4,) * 12,
        "ABOVE": ((6, 27, 21),) * 12,
        "BELOW": ((6, 27, 33),) * 12,
    }
    for name, actions in variants.items():
        run(env, name, actions)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
