import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import numpy as np
import gkm_arena as arena

import players


def push_until_blocked(env, action, limit=24):
    used = 0
    for _ in range(limit):
        before = np.asarray(env.frame()).copy()
        level = env.levels_completed
        env.step(action)
        if env.levels_completed != level:
            return used + 1, True
        if np.array_equal(before, np.asarray(env.frame())):
            return used, False
        used += 1
    return used, False


def place(env, turns):
    path = [5] * turns
    for _ in range(turns):
        env.step(5)
    up, won = push_until_blocked(env, 1)
    path += [1] * up
    if won:
        return path, True
    right, won = push_until_blocked(env, 4)
    path += [4] * right
    return path, won


def rotation_search(root, clicks, goal_level):
    tried = 0

    def visit(node, index, path, rotations):
        nonlocal tried
        if node.levels_completed >= goal_level:
            return path, rotations
        if index == len(clicks):
            tried += 1
            return None
        selected = node.clone()
        click = clicks[index]
        if click is not None:
            selected.step(6, *click)
        prefix = [] if click is None else [[6, *click]]
        for turns in range(5):
            child = selected.clone()
            placed, won = place(child, turns)
            result_path = path + prefix + placed
            if won or child.levels_completed >= goal_level:
                return result_path, rotations + [turns]
            found = visit(child, index + 1, result_path, rotations + [turns])
            if found is not None:
                return found
        return None

    found = visit(root, 0, [], [])
    print("search", "tried", tried, "found", found)
    return found


def probe(env):
    while env.levels_completed < 4:
        getattr(players, f"play_level_{env.levels_completed + 1}")(env)
    print("validate_level_5")
    rotation_search(env, [None, (54, 6), (5, 38), (47, 47)], 5)
    players.play_level_5(env)
    while env.levels_completed < 5:
        getattr(players, f"play_level_{env.levels_completed + 1}")(env)
    print("search_level_6")
    clicks = [None, (52, 10), (24, 27), (7, 47), (38, 52)]
    rotation_search(env, clicks, 6)


arena.run_program("cn04", probe)
