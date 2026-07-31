"""Bounded exhaustive test of the 13-action level-9 checkpoint budget."""

from collections import deque

import gkm_try

from perception import arr
from probe9_prefix_shortcuts import reach_level_9
from probe9_verify import boxes, target_state


def inspect(env):
    reach_level_9(env)
    base_level = env.levels_completed
    queue = deque([(env.clone(), ())])
    seen = {arr(env.frame()).tobytes()}
    transitions = 0
    deepest = 0
    best = None

    while queue and len(seen) < 20000:
        node, path = queue.popleft()
        deepest = max(deepest, len(path))
        if len(path) >= 13:
            continue
        for action in node.actions:
            child = node.clone()
            child.step(action)
            child_path = path + (int(action),)
            transitions += 1
            if child.levels_completed > base_level:
                print("BUDGET_BFS_WIN", child_path, transitions, flush=True)
                return
            target = target_state(child.frame())
            metric = (
                len(target["filled"]),
                not bool(boxes(child.frame(), 15)),
                -len(target["empty"]),
            )
            if best is None or metric > best[0]:
                best = (metric, child_path)
            key = arr(child.frame()).tobytes()
            if key not in seen:
                seen.add(key)
                queue.append((child, child_path))
            if len(seen) >= 20000:
                break

    print(
        "BUDGET_BFS_NONE",
        {
            "deepest": deepest,
            "states": len(seen),
            "transitions": transitions,
            "best": best,
        },
        flush=True,
    )


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
