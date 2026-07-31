import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import players


def board_key(env):
    return np.asarray(env.frame())[:55].tobytes()


def meter(env):
    frame = np.asarray(env.frame())
    return int(np.count_nonzero(frame[60:] == 11))


def lives(env):
    frame = np.asarray(env.frame())
    return int(np.count_nonzero(frame[61:63, 55:64] == 8))


def probe(env):
    players.play_level_1(env)
    players.play_level_2(env)
    players.play_level_3(env)
    base_level = int(env.levels_completed)
    start_lives = lives(env)
    queue = deque([(env.clone(), ())])
    best_meter = {board_key(env): meter(env)}
    expanded = 0
    path = None
    while queue and expanded < 1500:
        node, prefix = queue.popleft()
        if len(prefix) >= 90:
            continue
        expanded += 1
        for action in node.actions:
            child = node.clone()
            child.step(int(action))
            child_path = prefix + (int(action),)
            if int(child.levels_completed) > base_level:
                path = child_path
                queue.clear()
                break
            if child.terminal() or lives(child) != start_lives:
                continue
            key = board_key(child)
            remaining = meter(child)
            if remaining <= best_meter.get(key, -1):
                continue
            best_meter[key] = remaining
            queue.append((child, child_path))
    print(
        "search",
        "base",
        base_level,
        "expanded",
        expanded,
        "states",
        len(best_meter),
        "path",
        path,
    )
    if path is not None:
        result = env.clone()
        for action in path:
            result.step(action)
        print(
            "verified",
            int(result.levels_completed),
            "length",
            len(path),
            "terminal",
            bool(result.terminal()),
        )


arena.run_program("ls20", probe)
