"""Enumerate the ten ways both level-5 marker tiles can occupy its work area."""
from collections import deque
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from probe_l5 import ACTIONS, SLOTS, tile_state
from solve import solve


PERMS = (
    (1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
     18, 19, 20),
    (4, 0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
     18, 19, 20),
    (1, 2, 3, 4, 18, 0, 5, 6, 7, 8, 11, 12, 9, 10, 13, 14, 15, 16,
     19, 20, 17),
    (5, 0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15, 16, 17, 20,
     4, 18, 19),
)


def destinations(source_perm):
    result = [None] * len(source_perm)
    for destination, source in enumerate(source_perm):
        result[source] = destination
    return tuple(result)


MOVES = tuple(destinations(perm) for perm in PERMS)


def run(env):
    solve(env)
    base_level = env.levels_completed
    values = tile_state(env.frame())
    start = tuple(i for i, value in enumerate(values) if value == 11)
    queue = deque([(start, ())])
    seen = {start}
    completed = []
    while queue:
        state, path = queue.popleft()
        if all(position < 5 for position in state):
            completed.append((state, path))
        for op, move in enumerate(MOVES):
            child = tuple(sorted(move[position] for position in state))
            if child not in seen:
                seen.add(child)
                queue.append((child, path + (op,)))
    print("marker-start", start, "pair-states", len(seen),
          "completed-placements", len(completed))
    for state, path in sorted(completed, key=lambda item: len(item[1])):
        clone = env.clone()
        peak = 0
        for op in path:
            clone.step(*ACTIONS[op])
            peak = max(peak, sum(value == 11
                                 for value in tile_state(clone.frame())[:5]))
            if clone.levels_completed > base_level:
                print("FOUND", state, "path", path, "peak", peak)
                return
        print("checked", state, "depth", len(path), "peak", peak,
              "level", clone.levels_completed)
    print("no-marker-only-goal")


if __name__ == "__main__":
    A.run_program("lp85", run)
