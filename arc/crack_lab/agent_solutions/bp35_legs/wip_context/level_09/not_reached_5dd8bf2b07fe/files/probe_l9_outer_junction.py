"""Probe the switch junction at the twelfth outer descent."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls, enter_control_row


def enter_junction(env):
    enter_control_row(env)
    env.step(6, 9, 3)
    for _ in range(12):
        env.step(6, 3, 33)


def targets(env, color, areas):
    return [
        (6, round(blob.centroid[1]), round(blob.centroid[0]))
        for blob in connected_components(env.frame(), colors=(color,), min_area=3)
        if blob.bbox[0] < 63 and blob.area in areas
    ]


def report(label, env):
    print(
        label,
        compact(env),
        "controls",
        controls(env),
        "catches",
        targets(env, 15, (21,)),
        "blocks",
        targets(env, 14, (21,)),
    )


def probe(env):
    enter_junction(env)
    report("JUNCTION", env)
    for control in controls(env):
        child = env.clone()
        child.step(*control)
        report(("FLIP", control), child)
        for action in (3, 4):
            moved = child.clone()
            for count in range(1, 5):
                moved.step(action)
                report(("MOVE", control, action, count), moved)
                if moved.terminal():
                    break
        for target in targets(child, 15, (21,)):
            clicked = child.clone()
            clicked.step(*target)
            report(("CATCH", control, target), clicked)
    for action in (3, 4):
        child = env.clone()
        for count in range(1, 7):
            child.step(action)
            report(("EDGE", action, count), child)
            if child.terminal():
                break


if __name__ == "__main__":
    arena.run_program("bp35", probe)
