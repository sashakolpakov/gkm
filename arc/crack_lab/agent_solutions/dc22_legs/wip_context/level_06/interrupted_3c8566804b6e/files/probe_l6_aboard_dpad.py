"""Compose aboard key states with each remote D-pad control."""
from itertools import product
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_aboard_bfs import enter_overlap


DPAD = {
    "U": (6, 50, 34),
    "D": (6, 50, 40),
    "L": (6, 46, 36),
    "R": (6, 54, 36),
}


def avatar_pixels(env):
    frame = np.asarray(env.frame())
    rows, cols = np.where(frame[:62, :40] == 14)
    return tuple((int(row), int(col)) for row, col in zip(rows, cols))


def ring(env):
    return tuple(
        (blob.bbox, blob.area)
        for blob in perception.connected_components(
            env.frame(), colors=(8,), min_area=20
        )
        if blob.bbox[1] < 40
    )


def solid_exits(env):
    frame = np.asarray(env.frame())
    return tuple(
        (row, col)
        for row in range(0, 62, 2)
        for col in range(0, 40, 2)
        if np.all(frame[row:row + 2, col:col + 2] == 11)
    )


def observe(env):
    solve.solve(env)
    root = enter_overlap(env)
    base_level = root.levels_completed
    seen_key_states = set()
    tested = 0
    print(
        "ABOARD_DPAD_ROOT", avatar_pixels(root), ring(root), flush=True,
    )
    for length in range(5):
        for sequence in product((1, 2, 3, 4), repeat=length):
            staged = root.clone()
            for action in sequence:
                staged.step(action)
            state_key = np.asarray(staged.frame())[:63].tobytes()
            if state_key in seen_key_states:
                continue
            seen_key_states.add(state_key)
            before_ring = ring(staged)
            before = np.asarray(staged.frame())[:62, :40].copy()
            for name, control in DPAD.items():
                child = staged.clone()
                child.step(*control)
                after = np.asarray(child.frame())[:62, :40]
                changed = int(np.count_nonzero(before != after))
                tested += 1
                if (
                    changed
                    or ring(child) != before_ring
                    or child.levels_completed > base_level
                    or solid_exits(child)
                ):
                    print(
                        "ABOARD_DPAD_EFFECT", sequence, name,
                        "avatar", avatar_pixels(staged),
                        "to", avatar_pixels(child),
                        "ring", before_ring, "to", ring(child),
                        "delta", changed,
                        "exits", solid_exits(child),
                        "level", child.levels_completed,
                        flush=True,
                    )
    print(
        "ABOARD_DPAD_DONE", len(seen_key_states), tested, flush=True,
    )


arena.run_program("dc22", observe)
