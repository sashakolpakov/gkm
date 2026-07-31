"""Test coordinate selection of directions while the avatar boards the ring."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_aboard_bfs import enter_overlap


CASES = (
    ("UP", (6, 34, 56), (6, 50, 34)),
    ("DOWN", (6, 34, 60), (6, 50, 40)),
    ("LEFT", (6, 32, 58), (6, 46, 36)),
    ("RIGHT", (6, 36, 58), (6, 54, 36)),
)


def ring(env):
    return [
        blob.bbox
        for blob in perception.connected_components(
            env.frame(), colors=(8,), min_area=20
        )
        if blob.bbox[1] < 32
    ]


def avatar(env):
    return [
        (blob.bbox, blob.area)
        for blob in perception.connected_components(
            env.frame(), colors=(14,), min_area=2
        )
        if blob.bbox[1] < 40
    ]


def observe(env):
    solve.solve(env)
    root = enter_overlap(env)
    for name, selector_click, dpad_click in CASES:
        branch = root.clone()
        before_ring = ring(branch)
        branch.step(*selector_click)
        after_selector = ring(branch)
        branch.step(*dpad_click)
        print(
            "REMOTE_CLICK", name, before_ring, after_selector,
            ring(branch), avatar(branch), branch.levels_completed,
        )


arena.run_program("dc22", observe)
