"""Symbolic symmetry search and clone verification for lp85 level 7."""
from collections import deque
from collections import Counter
import itertools
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from solve import solve

CONTROLS = ((22, 34), (32, 42))
POSITIONS = (
    (23, 20), (23, 23), (23, 26), (23, 29),
    (23, 32), (23, 35), (23, 38), (23, 41),
    (26, 20), (29, 20),
    (35, 29), (35, 32), (38, 32), (38, 29),
)
PAIRS = (
    (0, 7), (1, 6), (2, 5), (3, 4),
    (0, 9),
    (10, 12), (11, 13),
)
# Destination-to-source permutations observed for the column cycle and for
# the simultaneous row/right-square rotations.
PERMS = (
    (9, 1, 2, 3, 4, 5, 6, 7, 0, 8, 10, 11, 12, 13),
    (7, 0, 1, 2, 3, 4, 5, 6, 8, 9, 13, 10, 11, 12),
)


def state(frame):
    f = np.asarray(frame)
    return tuple(int(f[p]) for p in POSITIONS)


def apply(values, perm):
    return tuple(values[source] for source in perm)


def score(values):
    return sum(values[a] == values[b] for a, b in PAIRS)


def equality_key(values):
    labels = {}
    return tuple(labels.setdefault(value, len(labels)) for value in values)


def inverse_perm(perm):
    out = [None] * len(perm)
    for destination, source in enumerate(perm):
        out[source] = destination
    return tuple(out)


def symmetric_states(start, square_groups):
    """Enumerate all arrangements with the same colors satisfying PAIRS."""
    groups = (
        (0, 7, 9), (1, 6), (2, 5), (3, 4),
    ) + square_groups + ((8,),)
    colors = tuple(sorted(set(start)))
    wanted = Counter(start)
    for assignment in itertools.product(colors, repeat=len(groups)):
        used = Counter()
        for color, group in zip(assignment, groups):
            used[color] += len(group)
        if used != wanted:
            continue
        values = [None] * len(start)
        for color, group in zip(assignment, groups):
            for position in group:
                values[position] = color
        yield tuple(values)


def run(env):
    solve(env)
    start = state(env.frame())
    square_variants = (
        ((10, 12), (11, 13)),  # 180-degree
        ((10, 11), (13, 12)),  # left-right reflection
        ((10, 13), (11, 12)),  # top-bottom reflection
    )
    goals = tuple({
        goal
        for square_groups in square_variants
        for goal in symmetric_states(start, square_groups)
    })
    queue = deque(goals)
    paths = {goal: () for goal in goals}
    inverses = tuple(inverse_perm(p) for p in PERMS)
    found = paths.get(start)
    while found is None and queue and len(paths) < 2000000:
        values = queue.popleft()
        path = paths[values]
        for op, inverse in enumerate(inverses):
            predecessor = apply(values, inverse)
            if predecessor in paths:
                continue
            paths[predecessor] = (op,) + path
            if predecessor == start:
                found = paths[predecessor]
                break
            queue.append(predecessor)
    print("result", found, "states", len(paths), "goals", len(goals))
    if found is None:
        return
    clone = env.clone()
    for op in found:
        clone.step(6, *CONTROLS[op])
    print("verified", clone.levels_completed, clone.terminal(),
          "score", score(state(clone.frame())), "state", state(clone.frame()))


if __name__ == "__main__":
    A.run_program("lp85", run)
