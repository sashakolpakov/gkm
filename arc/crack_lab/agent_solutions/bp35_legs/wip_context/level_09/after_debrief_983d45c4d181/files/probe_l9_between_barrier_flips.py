"""Reverse the retained switch at each step between the partial and full walls."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9
from probe_l9_supported_final_alignment import aligned


def boxes(env, color):
    return tuple(
        blob.bbox
        for blob in connected_components(env.frame(), colors=(color,), min_area=2)
        if blob.bbox[0] < 63
    )


def signature(env):
    return {
        "terminal": bool(env.terminal()),
        "levels": int(env.levels_completed),
        "avatar": boxes(env, 9),
        "controls": controls(env),
        "goals": boxes(env, 7),
        "grid": compact(env)["grid9"],
    }


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def probe(env):
    enter_level_9(env)
    for column in (5, 6, 7):
        x = 3 + 6 * column
        root = aligned(env, 5, column)
        for descent in range(7):
            if descent:
                root.step(6, x, 33)
            visible = controls(root)
            print("PRE", column, descent, signature(root), flush=True)
            if not visible or root.terminal():
                continue
            flipped = root.clone()
            flipped.step(*visible[-1])
            print("FLIP", column, descent, signature(flipped), flush=True)
            if flipped.terminal() or int(flipped.levels_completed) >= 9:
                continue
            for name, actions in (
                ("VERT", ((6, x, 33),) * 6),
                ("LEFT", (3,) * 3),
                ("RIGHT", (4,) * 3),
            ):
                child = flipped.clone()
                for action in actions:
                    step(child, action)
                    if child.terminal() or int(child.levels_completed) >= 9:
                        break
                print(name, column, descent, signature(child), flush=True)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
