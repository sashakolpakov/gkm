"""Bounded reward search over the four observed level-4 operations."""
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import arr, bounded_bfs, level_goal
from solve import solve


CLICKS = (
    (6, 15, 6),   # horizontal one way
    (6, 15, 25),  # horizontal inverse
    (6, 6, 15),   # vertical one way
    (6, 25, 15),  # vertical inverse
)


class CoordinateAction:
    def __init__(self, click):
        self.click = click


def run(env):
    solve(env)
    base_level = env.levels_completed
    start = time.monotonic()

    # Local BFS variant for coordinate actions, retaining shallow arena clones.
    queue = [(env.clone(), [])]
    seen = {arr(env.frame()).tobytes()}
    head = 0
    answer = None
    max_states = 12000
    max_depth = 14
    while head < len(queue) and len(seen) <= max_states:
        node, path = queue[head]
        head += 1
        if node.levels_completed > base_level:
            answer = path
            break
        if len(path) >= max_depth or node.terminal():
            continue
        for click in CLICKS:
            child = node.clone()
            child.step(*click)
            key = arr(child.frame()).tobytes()
            if key in seen:
                continue
            seen.add(key)
            child_path = path + [click]
            if child.levels_completed > base_level:
                answer = child_path
                queue = []
                break
            queue.append((child, child_path))
        if answer is not None:
            break
    print("search", "states", len(seen), "depth",
          max((len(path) for _, path in queue), default=0),
          "seconds", round(time.monotonic() - start, 2), "answer", answer)


if __name__ == "__main__":
    A.run_program("lp85", run)
