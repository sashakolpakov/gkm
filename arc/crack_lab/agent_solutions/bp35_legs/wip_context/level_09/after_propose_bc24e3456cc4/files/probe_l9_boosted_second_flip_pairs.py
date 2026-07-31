"""Test ordered pairs of the catch cells that affect the boosted landing."""

import sys
from itertools import product

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_boosted_gate_alignment import gate
from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_route_deletions import enter_level_9


TARGETS = (
    (21, 21),
    (21, 39),
    (27, 39),
    (21, 45),
    (27, 45),
    (21, 51),
    (27, 51),
    (21, 57),
    (27, 57),
    (33, 57),
)


def avatar_top(env):
    avatars = connected_components(env.frame(), colors=(9,), min_area=3)
    return avatars[0].bbox[0] if avatars else 99


def probe(env):
    enter_level_9(env)
    root = gate(env, 1)
    for first, second in product(TARGETS, repeat=2):
        child = root.clone()
        child.step(6, *first)
        child.step(6, *second)
        visible = controls(child)
        if child.terminal() or not visible:
            continue
        child.step(*visible[0])
        if (
            not child.terminal()
            and avatar_top(child) <= 30
            and len(controls(child)) >= 2
        ):
            print("FOUND", first, second, flush=True)
            report("FLIPPED", child)
            return
    print("NO_FOUND", len(TARGETS) ** 2, flush=True)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
