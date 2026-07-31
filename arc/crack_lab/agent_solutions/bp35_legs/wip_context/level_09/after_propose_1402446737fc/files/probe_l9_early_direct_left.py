"""Move left immediately after the earliest far-right gravity flip."""

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


def enter_early_left(env, steps=3):
    enter_early_flip(env, 0)
    for _ in range(steps):
        env.step(3)


def probe(env):
    enter_early_flip(env, 0)
    report("ENTRY", env)
    for step in range(1, 8):
        env.step(3)
        report(("LEFT", step), env)
        if env.terminal() or int(env.levels_completed) >= 9:
            return


if __name__ == "__main__":
    arena.run_program("bp35", probe)
