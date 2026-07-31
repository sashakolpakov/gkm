import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception
import players


TO_LEFT = (3, 3, 3, 2, 2, 2, 4, 1, 2, 3, 3, 1, 1, 1)
TRANSFER = (2, 2, 4, 1, 1, 1, 1, 4, 1, 4, 4, 4, 2, 2, 2, 3, 3, 1, 2)


def avatar_tile(env):
    pieces = perception.connected_components(env.frame(), colors=(12,), min_area=3)
    avatar = max(pieces, key=lambda piece: piece.area)
    row, col = avatar.centroid
    return round((row - 2) / 5), round((col - 1) / 5)


def meter(env):
    return int(np.count_nonzero(np.asarray(env.frame())[60:] == 11))


def lives(env):
    return int(np.count_nonzero(np.asarray(env.frame())[61:63, 55:64] == 8))


def state_key(env):
    frame = np.asarray(env.frame())
    board_tokens = tuple(map(tuple, np.argwhere(frame[:55] == 11)))
    return avatar_tile(env), board_tokens, glyph(frame)


def glyph(frame):
    sampled = np.asarray(frame)[55:61:2, 3:9:2]
    return tuple("".join(f"{int(value):X}" for value in row) for row in sampled)


def probe(env):
    players.play_level_1(env)
    players.play_level_2(env)
    players.play_level_3(env)
    for action in TO_LEFT:
        env.step(action)

    transfer = TRANSFER
    print("transfer", transfer)
    node = env.clone()
    for action in transfer:
        node.step(action)
    print(
        "contact1",
        "level",
        int(node.levels_completed),
        "meter",
        meter(node),
        "glyph",
        glyph(node.frame()),
    )
    for action in (1, 2, 1, 2):
        node.step(action)
        print(
            "cycle",
            action,
            "level",
            int(node.levels_completed),
            "meter",
            meter(node),
            "glyph",
            glyph(node.frame()),
        )
    for action in node.actions:
        child = node.clone()
        child.step(int(action))
        print(
            "matched_action",
            int(action),
            "level",
            int(child.levels_completed),
            "meter",
            meter(child),
            "glyph",
            glyph(child.frame()),
        )
    start_lives = lives(node)
    queue = deque([(node.clone(), ())])
    best = {state_key(node): meter(node)}
    finish = None
    expanded = 0
    while queue and expanded < 500:
        search_node, path = queue.popleft()
        if len(path) >= 40:
            continue
        expanded += 1
        for action in search_node.actions:
            child = search_node.clone()
            child.step(int(action))
            child_path = path + (int(action),)
            if int(child.levels_completed) > 3:
                finish = child_path
                queue.clear()
                break
            if child.terminal() or lives(child) != start_lives:
                continue
            key = state_key(child)
            remaining = meter(child)
            if remaining <= best.get(key, -1):
                continue
            best[key] = remaining
            queue.append((child, child_path))
    print("finish", finish, "expanded", expanded, "states", len(best))


arena.run_program("ls20", probe)
