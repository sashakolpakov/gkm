"""Test ring-direction controls from colored physical-world target cells."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_exact_crossings import placements_with_paths
from probe_l6_reach_by_ring import vertical_entry
from probe_l6_right import MAIN, avatar_position, enter_right


CONTROLS = {
    "UP": (6, 50, 34),
    "DOWN": (6, 50, 40),
    "LEFT": (6, 46, 36),
    "RIGHT": (6, 54, 36),
}


def ring(env):
    return tuple(
        blob.bbox
        for blob in perception.connected_components(
            env.frame(), colors=(8,), min_area=20
        )
        if blob.bbox[1] < 32
    )


def test(label, node, base_level):
    for name, control in CONTROLS.items():
        branch = node.clone()
        before = ring(branch)
        branch.step(*control)
        after = ring(branch)
        print(
            "PHYSICAL_CONTROL", label, avatar_position(node),
            name, before, after,
            branch.levels_completed - base_level,
        )


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    placement = placements_with_paths(enter_right(env, 3))[0][0]
    lower = vertical_entry(placement)
    vertical_center = lower.clone()
    for _ in range(12):
        vertical_center.step(1)
    horizontal_center = vertical_center.clone()
    horizontal_center.step(*MAIN)
    upper_target = lower.clone()
    for _ in range(19):
        upper_target.step(1)
    test("lower", lower, base_level)
    test("vertical_center", vertical_center, base_level)
    test("horizontal_center", horizontal_center, base_level)
    test("upper_target", upper_target, base_level)


arena.run_program("dc22", observe)
