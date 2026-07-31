"""Check all world goal tiles after the ring reaches its top terminal."""
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import solve
from probe_l6_cargo_top import CARGO_TOP_PATH
from probe_l6_right import MAIN, SELECTOR, enter_right


def apply(env, path):
    for action in path:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def color11_blocks(env):
    frame = np.asarray(env.frame())
    blocks = []
    for row in range(31):
        for col in range(20):
            block = frame[
                2 * row:2 * row + 2,
                2 * col:2 * col + 2,
            ]
            count = int(np.count_nonzero(block == 11))
            if count:
                blocks.append((count, row, col))
    return sorted(blocks, reverse=True)


def observe(env):
    solve.solve(env)
    node = enter_right(env, 3)
    apply(node, CARGO_TOP_PATH)
    print("TOP_RING_GOALS", color11_blocks(node)[:12], flush=True)
    node.step(2)
    node.step(*MAIN)
    for selector_phase in range(4):
        branch = node.clone()
        apply(branch, [SELECTOR] * selector_phase + [MAIN])
        print(
            "TOP_RING_DEST", selector_phase,
            color11_blocks(branch)[:12],
            branch.levels_completed, flush=True,
        )
        for main_phase in range(1, 5):
            branch.step(*MAIN)
            print(
                "TOP_RING_DEST_MAIN", selector_phase, main_phase,
                color11_blocks(branch)[:12],
                branch.levels_completed, flush=True,
            )


arena.run_program("dc22", observe)
