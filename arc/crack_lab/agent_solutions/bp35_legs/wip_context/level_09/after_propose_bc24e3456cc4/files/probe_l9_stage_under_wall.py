"""Stage the lower catch beneath the trapdoor outside the solid wall."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_second_descent import component_at
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
        "terminal",
        bool(env.terminal()),
        "controls",
        controls(env),
        "goals",
        goals(env),
    )


def enter_wall_outside(env):
    enter_second_gap(env, 2)
    for _ in range(6):
        env.step(6, 15, 35)
    env.step(6, 45, 35)
    for col in (6, 7, 8):
        env.step(6, 3 + 6 * col, 41)
    for col in range(3, 10):
        env.step(6, 3 + 6 * col, 27)
        env.step(4)
    env.step(6, 57, 35)


def probe(env):
    enter_second_gap(env, 2)
    for _ in range(6):
        env.step(6, 15, 35)
    report("SHELF", env)
    env.step(6, 45, 35)
    report("PLATFORM_ON", env)
    for col in (6, 7, 8):
        env.step(6, 3 + 6 * col, 41)
        report(("STAGE_CATCH", col), env)
    for col in range(3, 10):
        env.step(6, 3 + 6 * col, 27)
        env.step(4)
        report(("HANDOFF", col), env)
        if env.terminal():
            return
    env.step(6, 57, 35)
    report(
        (
            "OPEN_DOOR",
            "under",
            component_at(env, 57, 35),
        ),
        env,
    )
    for depth in range(1, 16):
        under = component_at(env, 57, 35)
        if not under or under[0] != 15 or under[1] != 21:
            print("STOP", depth - 1, under)
            return
        env.step(6, 57, 35)
        report(("DESCEND", depth, component_at(env, 57, 35)), env)
        if env.terminal() or int(env.levels_completed) >= 9:
            return


if __name__ == "__main__":
    arena.run_program("bp35", probe)
