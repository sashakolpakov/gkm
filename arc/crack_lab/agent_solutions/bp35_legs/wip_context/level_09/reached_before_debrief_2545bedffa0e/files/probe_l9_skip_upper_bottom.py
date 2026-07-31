"""Test the lower-wall handoff at the end of the fast c4 descent."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9
from probe_l9_skip_upper_drop import DROP, dropped


def result(env):
    goals = tuple(
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(7,), min_area=2)
        if blob.bbox[0] < 63
    )
    avatars = tuple(
        blob.bbox
        for blob in connected_components(env.frame(), colors=(9, 11), min_area=2)
        if blob.bbox[0] < 63
    )
    return (
        bool(env.terminal()),
        int(env.levels_completed) + 1,
        controls(env),
        avatars,
        goals,
        compact(env)["grid9"],
    )


def run(root, depth, name, actions):
    child = dropped(root, depth)
    print(depth, name, 0, result(child), flush=True)
    for index, action in enumerate(actions, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        print(depth, name, index, action, result(child), flush=True)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_level_9(env)
    variants = {
        "CLEAR_RIGHT": ((6, 33, 27), 4, (6, 33, 33), 4),
        "CLEAR_RIGHT_DROP": ((6, 33, 27), 4, (6, 33, 27), (6, 33, 33)),
        "REMOTE_RIGHT": ((6, 33, 21), (6, 33, 27), 4, (6, 33, 33)),
        "CLEAR_LEFT": ((6, 21, 27), 3, (6, 21, 33), 3),
        "MOVE_RIGHT": (4, 4, (6, 33, 27), 4),
    }
    for depth in (8, 9):
        for name, actions in variants.items():
            run(env, depth, name, actions)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
