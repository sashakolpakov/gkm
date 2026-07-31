"""Map the deep chamber reached by omitting several opening climb clicks."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9, replay, route
from probe_l9_twelve_fast_frontier import SKIPS


def root_for(root, count):
    return replay(
        root,
        route(),
        skips=SKIPS | set(range(11, 11 + count)),
    )


def pieces(env):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            env.frame(), colors=(7, 8, 9, 11, 12, 14, 15), min_area=2
        )
        if blob.bbox[0] < 63
        and (blob.color in (7, 8, 9, 11, 12, 14) or blob.area == 21)
    )


def brief(env):
    return {
        "terminal": bool(env.terminal()),
        "level": int(env.levels_completed) + 1,
        "controls": controls(env),
        "grid": compact(env)["grid9"],
        "pieces": pieces(env),
    }


def run(root, count, name, actions):
    child = root_for(root, count)
    print(count, name, 0, brief(child), flush=True)
    for index, action in enumerate(actions, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        print(count, name, index, action, brief(child), flush=True)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    for count in (6, 10):
        run(env, count, "LEFT", (3,) * 10)
        run(env, count, "RIGHT", (4,) * 10)
    variants = {
        "CLEAR_C8_LEFT": ((6, 51, 27),) + (3,) * 10,
        "CLEAR_UPPER_LEFT": (
            (6, 51, 21),
            (6, 45, 21),
            (6, 39, 21),
            (6, 51, 27),
        )
        + (3,) * 10,
        "BELOW_C3": ((6, 21, 45), (6, 21, 45), 3, 4),
        "EDGE_RIGHT": (4, (6, 57, 27), 4, 4),
    }
    for name, actions in variants.items():
        run(env, 10, name, actions)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
