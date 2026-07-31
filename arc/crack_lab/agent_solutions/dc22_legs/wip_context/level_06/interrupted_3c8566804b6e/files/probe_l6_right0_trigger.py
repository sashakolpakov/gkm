"""Visit both halves of the cyan remote glyph before testing selector state 1."""
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_right import (
    MAIN,
    SELECTOR,
    avatar_position,
    enter_right,
)


def glyph(env):
    return perception.arr(env.frame())[48:50, 34:38].tolist()


def exact_walk(root):
    base_level = root.levels_completed
    queue = deque([(root.clone(), [])])
    seen = {perception.arr(root.frame())[:63].tobytes()}
    while queue:
        node, path = queue.popleft()
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            child_path = path + [action]
            if child.levels_completed > base_level:
                return len(seen), child_path
            key = perception.arr(child.frame())[:63].tobytes()
            if key in seen:
                continue
            seen.add(key)
            queue.append((child, child_path))
    return len(seen), None


def observe(env):
    solve.solve(env)
    node = enter_right(env, 0)
    before = glyph(node)
    # This loop consumes the entire glyph and returns to the portal pad.
    for action in (1, 1, 4, 2, 2, 3):
        node.step(action)
    print("RIGHT0_GLYPH", before, glyph(node), avatar_position(node))
    for selector_steps in range(4):
        branch = node.clone()
        branch.step(*MAIN)
        for _ in range(selector_steps):
            branch.step(*SELECTOR)
        branch.step(*MAIN)
        states, win = exact_walk(branch)
        print(
            "RIGHT0_FOLLOWUP", selector_steps,
            avatar_position(branch), states, win,
            branch.levels_completed,
        )


arena.run_program("dc22", observe)
