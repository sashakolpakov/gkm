"""Inspect marked work windows and infer their rewarded contents."""
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import connected_components


def tiles(frame, size):
    result = []
    for b in connected_components(frame, min_area=size * size):
        if b.area == size * size and b.size == (size, size):
            result.append((b.bbox[0], b.bbox[1], b.color))
    return tuple(result)


def run(env):
    states = []
    for clicks in range(5):
        node = env.clone()
        for _ in range(clicks):
            node.step(6, 5, 32)
        states.append(tiles(node.frame(), 4))
        print("L1", clicks, "level", node.levels_completed,
              "tiles", states[-1])
    # Infer the destination-source permutation from the non-rewarding samples.
    positions = tuple((r, c) for r, c, _ in states[0])
    values = tuple(
        tuple(dict(((r, c), v) for r, c, v in state)[p] for p in positions)
        for state in states
    )
    perm = []
    for destination in range(len(positions)):
        choices = [
            source for source in range(len(positions))
            if all(after[destination] == before[source]
                   for before, after in zip(values, values[1:]))
        ]
        perm.append(choices)
    predicted = tuple(values[-1][choice[0]] if len(choice) == 1 else None
                      for choice in perm)
    print("L1 positions", positions)
    print("L1 permutation", perm)
    print("L1 predicted-reward", predicted)


if __name__ == "__main__":
    A.run_program("lp85", run)
