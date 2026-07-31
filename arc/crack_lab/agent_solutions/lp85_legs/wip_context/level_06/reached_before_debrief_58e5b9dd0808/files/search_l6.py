"""Bounded clone BFS over level-6 responsive coordinate controls."""
from collections import deque
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from probe_l6 import CONTROLS, token_state
from solve import solve


def run(env):
    solve(env)
    base_level = env.levels_completed
    root = env.clone()
    positions = tuple(sorted(token_state(root.frame())))

    def key(node):
        frame = node.frame()
        return tuple(int(frame[r][c]) for r, c in positions)

    queue = deque([(root, ())])
    seen = {key(root)}
    last_depth = -1
    while queue and len(seen) <= 20000:
        node, path = queue.popleft()
        if len(path) != last_depth:
            last_depth = len(path)
            print("depth", last_depth, "seen", len(seen),
                  "queue", len(queue), flush=True)
        if len(path) >= 18:
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
    print("run-result", A.run_program("lp85", run))
