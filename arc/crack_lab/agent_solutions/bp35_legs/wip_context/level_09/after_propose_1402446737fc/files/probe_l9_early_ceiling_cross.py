"""Cross left by growing a ceiling from same-row catches at height three."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_early_flip_climb import enter_early_flip


def report(label, env):
    avatars = [
        blob.bbox
        for blob in connected_components(env.frame(), colors=(9, 11), min_area=3)
        if blob.bbox[0] < 63
    ]
    goals = [
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(7,), min_area=3)
        if blob.bbox[0] < 63
    ]
    print(
        label,
        compact(env),
        "terminal",
        bool(env.terminal()),
        "avatars",
        avatars,
        "controls",
        controls(env),
        "goals",
        goals,
    )


def enter_early_ceiling_left(env, stop_col=2, height=1):
    enter_early_flip(env, height)
    for col in range(8, stop_col - 1, -1):
        env.step(6, 3 + 6 * col, 39)
        env.step(3)


def probe(env):
    base = env.clone()
    for height in (0, 1):
        child = base.clone()
        enter_early_flip(child, height)
        report(("ENTRY", height), child)
        for col in range(8, 1, -1):
            child.step(6, 3 + 6 * col, 39)
            report(("GROW_CEILING", height, col), child)
            if child.terminal():
                break
            child.step(3)
            report(("MOVE_LEFT", height, col), child)
            if child.terminal() or int(child.levels_completed) >= 9:
                break


if __name__ == "__main__":
    arena.run_program("bp35", probe)
