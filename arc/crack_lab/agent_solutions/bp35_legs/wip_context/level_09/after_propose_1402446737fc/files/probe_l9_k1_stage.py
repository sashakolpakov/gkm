"""Stage the K-room catch remotely before attempting the second ascent."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _cell_shape, band_grid, click_action, moves_used
from perception import color_counts, connected_components
from probe_l9_k_room import enter_k_room


def compact(env):
    blobs = connected_components(env.frame(), colors=(7, 8, 9, 12, 14, 15), min_area=3)
    return {
        "level": int(env.levels_completed) + 1,
        "terminal": bool(env.terminal()),
        "moves": moves_used(env.frame()),
        "colors": color_counts(env.frame()),
        "grid": tuple("".join(row) for row in band_grid(env.frame())),
        "shape4": tuple(_cell_shape(env.frame(), 4, col) for col in range(8)),
        "shape5": tuple(_cell_shape(env.frame(), 5, col) for col in range(8)),
        "pieces": tuple(
            (b.color, b.bbox, b.area)
            for b in blobs
            if b.bbox[0] < 63 and b.color in (7, 8, 9, 12, 14)
        ),
    }


def run(root, name, route):
    child = root.clone()
    print("START", name, compact(child))
    for index, action in enumerate(route, 1):
        child.step(*action) if isinstance(action, tuple) else child.step(action)
        print("STEP", name, index, action, compact(child))
        if child.terminal() or int(child.levels_completed) >= 9:
            break


def probe(env):
    enter_k_room(env)
    env.step(*click_action(5, 5))
    tests = {
        "left_remote_climb": [
            click_action(6, 4), 3,
            click_action(5, 5),
            click_action(5, 4),
        ],
        "right_remote_climb": [
            click_action(6, 6), 4,
            click_action(5, 5),
            click_action(5, 6),
        ],
        "left2_stage_climb": [
            click_action(6, 4), 3,
            click_action(6, 3), 3,
            click_action(5, 4),
            click_action(5, 3),
        ],
    }
    for name, route in tests.items():
        run(env, name, route)


arena.run_program("bp35", probe)
