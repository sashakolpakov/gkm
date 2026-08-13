"""Test valid in-frame header coordinates for a public interaction."""

import gkm_try
from perception import arr


def probe(env):
    root = arr(env.frame())[1:, :].tobytes(); changed = []
    for y in (0, 1):
        for x in range(64):
            node = env.clone(); node.step(6, x, y)
            after = arr(node.frame())[1:, :].tobytes()
            if after != root or node.levels_completed != env.levels_completed:
                changed.append((x, y, node.levels_completed, after != root))
    print("HEADER_COORDINATES", changed)


gkm_try.A.run_program("lf52", probe)
