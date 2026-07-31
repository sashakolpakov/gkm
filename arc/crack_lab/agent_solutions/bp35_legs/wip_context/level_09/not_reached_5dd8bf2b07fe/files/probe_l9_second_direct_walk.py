"""Walk across the second control chamber after its immediate flip."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_second_control_chamber import enter_second_control


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


def enter_second_walk(env, steps=7):
    enter_second_control(env)
    env.step(*controls(env)[0])
    for _ in range(steps):
        env.step(4)


def probe(env):
    enter_second_control(env)
    env.step(*controls(env)[0])
    report("FLIPPED", env)
    for step in range(1, 11):
        env.step(4)
        report(("RIGHT", step), env)
        if env.terminal() or int(env.levels_completed) >= 9:
            return


if __name__ == "__main__":
    arena.run_program("bp35", probe)
