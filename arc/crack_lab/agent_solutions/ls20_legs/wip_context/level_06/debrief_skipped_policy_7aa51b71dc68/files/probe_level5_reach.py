"""Bounded position-graph probe for level 5."""
from collections import deque
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import connected_components
from probe_level5 import reach_level_5


DELTAS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}


def player_cell(env):
    nines = connected_components(env.frame(), colors=(9,), min_area=3)
    body = max(nines, key=lambda blob: blob.area)
    return ((body.bbox[0] - 2) // 5, (body.bbox[1] - 4) // 5)


def refill_cells(env):
    return tuple(
        ((blob.bbox[0] - 1) // 5, (blob.bbox[1] - 5) // 5)
        for blob in connected_components(env.frame(), colors=(11,), min_area=4)
        if blob.bbox[0] < 60
    )


def inspect(env):
    reach_level_5(env)
    root = env.clone()
    start = player_cell(root)
    queue = deque([(root, ())])
    seen = {start: ()}
    transitions = {}
    pickups = []
    while queue:
        node, path = queue.popleft()
        source = player_cell(node)
        before_refills = refill_cells(node)
        for action, delta in DELTAS.items():
            child = node.clone()
            child.step(action)
            target = player_cell(child)
            transitions[(source, action)] = target
            after_refills = refill_cells(child)
            if after_refills != before_refills:
                pickups.append((source, action, target, before_refills, after_refills))
            if (
                target not in seen
                and len(path) < 20
                and child.levels_completed == 4
            ):
                seen[target] = path + (action,)
                queue.append((child, path + (action,)))
    print("start", start, "reachable", len(seen))
    for row in range(12):
        print(
            f"{row:02}",
            " ".join(
                "##" if (row, col) not in seen else f"{len(seen[(row, col)]):02}"
                for col in range(12)
            ),
        )
    anomalies = []
    for (source, action), target in sorted(transitions.items()):
        dr, dc = DELTAS[action]
        expected = (source[0] + dr, source[1] + dc)
        if target not in (source, expected):
            anomalies.append((source, action, expected, target))
    print("portals", anomalies)
    print("pickups", pickups)
    print("paths", sorted(seen.items()))


if __name__ == "__main__":
    levels, path, error = A.run_program("ls20", inspect)
    print("probe_result", levels, len(path), error)
