"""Route across the first interior shelf to its single opening."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_second_gap_cross import enter_second_gap


def goals(env):
    return [
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(7,), min_area=3)
        if blob.bbox[0] < 63
    ]


def report(label, env):
    print(
        label,
        compact(env),
        "controls",
        controls(env),
        "goals",
        goals(env),
    )


def enter_interior_gap1(env):
    enter_second_gap(env, 2)
    for _ in range(6):
        env.step(6, 15, 35)
    for col in (7,):
        env.step(6, 3 + 6 * col, 35)
    for col in range(3, 8):
        env.step(6, 3 + 6 * col, 27)
        env.step(4)


def probe(env):
    enter_second_gap(env, 2)
    for _ in range(6):
        env.step(6, 15, 35)
    report("SHELF", env)
    for col in (7,):
        env.step(6, 3 + 6 * col, 35)
        report(("STAGE", col), env)
    for col in range(3, 8):
        env.step(6, 3 + 6 * col, 27)
        env.step(4)
        report(("HANDOFF", col), env)
        if env.terminal():
            return
    for depth in range(1, 16):
        env.step(6, 45, 35)
        report(("DESCEND", depth), env)
        if env.terminal() or int(env.levels_completed) >= 9:
            return


if __name__ == "__main__":
    arena.run_program("bp35", probe)
