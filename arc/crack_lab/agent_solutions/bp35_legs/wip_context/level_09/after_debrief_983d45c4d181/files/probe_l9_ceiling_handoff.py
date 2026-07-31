"""Stage a ceiling and same-row catch before moving under reversed gravity."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_after_wall_flip import enter_after_wall_flip


def report(label, env):
    avatars = [
        blob.bbox
        for blob in connected_components(env.frame(), colors=(9, 11), min_area=3)
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
    )


def enter_ceiling_lane(env, stop_col=8):
    enter_after_wall_flip(env)
    for col in range(8, stop_col - 1, -1):
        x = 3 + 6 * col
        env.step(6, x, 33)
        env.step(6, x, 39)
        env.step(3)


def probe(env):
    enter_after_wall_flip(env)
    report("ENTRY", env)
    for col in range(8, 2, -1):
        x = 3 + 6 * col
        env.step(6, x, 33)
        report(("OPEN_CEILING", col), env)
        if env.terminal():
            return
        env.step(6, x, 39)
        report(("RESTORE_CEILING", col), env)
        if env.terminal():
            return
        env.step(3)
        report(("MOVE_LEFT", col), env)
        if env.terminal():
            return


if __name__ == "__main__":
    arena.run_program("bp35", probe)
