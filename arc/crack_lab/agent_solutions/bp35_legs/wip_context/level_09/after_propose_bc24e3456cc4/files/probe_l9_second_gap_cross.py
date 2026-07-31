"""Cross the shaft wall when the staged catch row aligns after one descent."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_gap_flip import enter_gap_landing


def goals(env):
    return [
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(7,), min_area=3)
        if blob.bbox[0] < 63
    ]


def report(label, env):
    avatars = [
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(9, 11), min_area=3)
        if blob.bbox[0] < 63
    ]
    print(
        label,
        compact(env),
        "avatars",
        avatars,
        "controls",
        controls(env),
        "goals",
        goals(env),
    )


def enter_second_gap(env, stop_col=1):
    enter_gap_landing(env)
    env.step(6, 3, 35)
    for col in range(1, stop_col + 1):
        env.step(6, 3 + 6 * col, 27)
        env.step(4)


def probe(env):
    enter_gap_landing(env)
    env.step(6, 3, 35)
    report("ALIGNED", env)
    for col in range(1, 10):
        env.step(6, 3 + 6 * col, 27)
        env.step(4)
        report(("HANDOFF", col), env)
        if env.terminal() or int(env.levels_completed) >= 9:
            return


if __name__ == "__main__":
    arena.run_program("bp35", probe)
