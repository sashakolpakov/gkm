"""Test direct walking across the fully restored first shelf."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact
from probe_l9_second_gap_cross import enter_second_gap


def report(label, env):
    avatars = [
        blob.bbox
        for blob in connected_components(env.frame(), colors=(9, 11), min_area=3)
        if blob.bbox[0] < 63
    ]
    print(label, compact(env), "terminal", bool(env.terminal()), "avatars", avatars)


def enter_staged_shelf(env):
    enter_second_gap(env, 2)
    for _ in range(6):
        env.step(6, 15, 35)
    env.step(6, 45, 35)
    for col in (6, 7, 8):
        env.step(6, 3 + 6 * col, 41)


def probe(env):
    enter_staged_shelf(env)
    report("ENTRY", env)
    for step in range(1, 8):
        env.step(4)
        report(("RIGHT", step), env)
        if env.terminal():
            return


if __name__ == "__main__":
    arena.run_program("bp35", probe)
