"""Bounded reward search over level-8's four observed controls."""
from collections import deque
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from solve import solve

CONTROLS = ((50, 24), (50, 29), (50, 34), (31, 57))


def run(env):
    solve(env)
    base_level = env.levels_completed
    # All persistent tokens lie at these sampled lattice coordinates.
    positions = tuple(
        (y, x)
        for y in range(6, 52, 3)
        for x in range(3, 43, 3)
    )

    def key(node):
        frame = node.frame()
        return bytes(int(frame[y][x]) for y, x in positions)

    queue = deque([(env.clone(), ())])
    seen = {key(queue[0][0])}
    last_depth = -1
    while queue and len(seen) < 30000:
        node, path = queue.popleft()
        if len(path) != last_depth:
            last_depth = len(path)
            print("depth", last_depth, "seen", len(seen),
                  "queue", len(queue), flush=True)
        if len(path) >= 22:
            continue
        for op, (x, y) in enumerate(CONTROLS):
            child = node.clone()
            child.step(6, x, y)
            child_path = path + (op,)
            if child.levels_completed > base_level:
                print("FOUND", child_path,
                      [CONTROLS[i] for i in child_path], flush=True)
                return
            child_key = key(child)
            if child_key not in seen and not child.terminal():
                seen.add(child_key)
                queue.append((child, child_path))
    print("stopped", len(seen), "queue", len(queue), flush=True)


if __name__ == "__main__":
    A.run_program("lp85", run)
