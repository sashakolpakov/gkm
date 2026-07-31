"""Enter the c5-c6 wall opening from the early ten-skip frontier."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9, route, step
from probe_l9_twelve_fast_frontier import SKIPS


OMITTED = SKIPS | set(range(11, 21))


def frontier(root):
    child = root.clone()
    for index, (_, action) in enumerate(route()):
        if index > 33:
            break
        if index not in OMITTED:
            step(child, action)
    return child


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


def run(root, name, actions):
    child = frontier(root)
    report((name, 0), child)
    for index, action in enumerate(actions, 1):
        step(child, action)
        report((name, index, action), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    variants = {
        "C6_DROP": (
            3,
            3,
            (6, 39, 27),
            3,
            *((6, 39, 33),) * 7,
        ),
        "C5_DROP": (
            3,
            3,
            (6, 39, 27),
            3,
            (6, 33, 27),
            3,
            *((6, 33, 33),) * 7,
        ),
        "C6_CLEAR_BELOW": (
            3,
            3,
            (6, 39, 27),
            3,
            (6, 39, 33),
            3,
            4,
        ),
    }
    for name, actions in variants.items():
        run(env, name, actions)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
