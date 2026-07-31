"""Bounded full-state BFS from the unlocked hub."""
from collections import deque
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_right import (
    MAIN,
    SELECTOR,
    TOP,
    avatar_position,
    enter_right,
)


DPAD_BY_POSITION = {
    (56, 34): (6, 50, 34),
    (60, 34): (6, 50, 40),
    (58, 32): (6, 46, 36),
    (58, 36): (6, 54, 36),
}


def key(env, selector_phase):
    return perception.arr(env.frame())[:63].tobytes(), selector_phase


def action_set(env):
    actions = [1, 2, 3, 4, TOP, MAIN, SELECTOR]
    dpad = DPAD_BY_POSITION.get(avatar_position(env))
    if dpad is not None:
        actions.append(dpad)
    return actions


def observe(env):
    solve.solve(env)
    root = enter_right(env, 3)
    root.step(*MAIN)
    base_level = root.levels_completed
    selector_phase = 3
    queue = deque([(root.clone(), [], selector_phase)])
    seen = {key(root, selector_phase)}
    print("FULL_BFS_START", avatar_position(root), base_level)
    while queue and len(seen) < 1500:
        node, path, phase = queue.popleft()
        if len(path) >= 90 or node.terminal():
            continue
        for action in action_set(node):
            child = node.clone()
            if isinstance(action, tuple):
                child.step(*action)
            else:
                child.step(action)
            child_path = path + [action]
            child_phase = (phase + 1) % 4 if action == SELECTOR else phase
            if child.levels_completed > base_level:
                print("FULL_BFS_WIN", len(child_path), child_path)
                return
            child_key = key(child, child_phase)
            if child_key in seen:
                continue
            seen.add(child_key)
            if len(seen) % 100 == 0:
                print(
                    "FULL_BFS_PROGRESS", len(seen), len(child_path),
                    avatar_position(child),
                )
            queue.append((child, child_path, child_phase))
    print("FULL_BFS_DONE", len(seen), len(queue))


arena.run_program("dc22", observe)
