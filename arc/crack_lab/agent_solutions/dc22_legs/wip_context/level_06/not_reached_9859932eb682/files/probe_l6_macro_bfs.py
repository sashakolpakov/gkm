"""Control-level BFS with exact movement closures."""
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


def frame_key(env):
    return perception.arr(env.frame())[:63].tobytes()


def state_key(env, selector_phase):
    return frame_key(env), selector_phase


def movement_closure(root, base_level):
    queue = deque([(root.clone(), [])])
    seen = {frame_key(root)}
    states = []
    while queue:
        node, path = queue.popleft()
        states.append((node, path))
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            child_path = path + [action]
            if child.levels_completed > base_level:
                return states, child_path
            child_key = frame_key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            queue.append((child, child_path))
    return states, None


def controls(node):
    out = [MAIN, TOP, SELECTOR]
    dpad = DPAD_BY_POSITION.get(avatar_position(node))
    if dpad is not None:
        out.insert(0, dpad)
    return out


def observe(env):
    solve.solve(env)
    root = enter_right(env, 3)
    root.step(*MAIN)
    base_level = root.levels_completed
    initial_phase = 3
    queue = deque([(root.clone(), [], initial_phase, 0)])
    seen = {state_key(root, initial_phase)}
    print("MACRO_BFS_START", avatar_position(root), base_level)
    while queue and len(seen) < 500:
        node, path, phase, control_depth = queue.popleft()
        if control_depth >= 35 or len(path) >= 140:
            continue
        walk_states, win_walk = movement_closure(node, base_level)
        if win_walk is not None:
            print("MACRO_BFS_WIN", len(path + win_walk), path + win_walk)
            return
        local_results = set()
        for walked, walk_path in walk_states:
            for control in controls(walked):
                child = walked.clone()
                child.step(*control)
                child_path = path + walk_path + [control]
                child_phase = (
                    (phase + 1) % 4 if control == SELECTOR else phase
                )
                if child.levels_completed > base_level:
                    print("MACRO_BFS_WIN", len(child_path), child_path)
                    return
                child_key = state_key(child, child_phase)
                if child_key in seen or child_key in local_results:
                    continue
                local_results.add(child_key)
                seen.add(child_key)
                queue.append(
                    (child, child_path, child_phase, control_depth + 1)
                )
                if len(seen) % 25 == 0:
                    print(
                        "MACRO_BFS_PROGRESS", len(seen),
                        control_depth + 1, len(child_path),
                        avatar_position(child),
                    )
                if len(seen) >= 500:
                    break
            if len(seen) >= 500:
                break
    print("MACRO_BFS_DONE", len(seen), len(queue))


arena.run_program("dc22", observe)
