import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception
import players


def avatar_tile(env):
    pieces = perception.connected_components(
        env.frame(), colors=(12,), min_area=3
    )
    avatar = max(pieces, key=lambda piece: piece.area)
    row, col = avatar.centroid
    return round((row - 2) / 5), round((col - 1) / 5)


def probe(env):
    players.play_level_1(env)
    players.play_level_2(env)
    players.play_level_3(env)
    start = env.clone()
    start_pos = avatar_tile(start)
    queue = deque([(start, (), start_pos)])
    paths = {start_pos: ()}
    specials = []

    while queue:
        node, path, pos = queue.popleft()
        if len(path) >= 35:
            continue
        before = np.asarray(node.frame())[:60].copy()
        for action in node.actions:
            child = node.clone()
            child.step(int(action))
            after = np.asarray(child.frame())[:60]
            new_pos = avatar_tile(child)
            board_delta = int(np.count_nonzero(before != after))
            displacement = abs(pos[0] - new_pos[0]) + abs(pos[1] - new_pos[1])
            if (
                board_delta not in (0, 50)
                or displacement not in (0, 1)
                or int(child.levels_completed) != 3
            ):
                specials.append(
                    (
                        pos,
                        int(action),
                        new_pos,
                        board_delta,
                        int(child.levels_completed),
                        path + (int(action),),
                    )
                )
            if new_pos not in paths:
                new_path = path + (int(action),)
                paths[new_pos] = new_path
                queue.append((child, new_path, new_pos))

    print("reachable", len(paths), "start", start_pos)
    for row in range(12):
        print(
            "".join(
                "." if (row, col) not in paths else f"{len(paths[(row, col)]) % 16:X}"
                for col in range(12)
            )
        )
    print("specials")
    for item in specials:
        print(item)
    farthest = max(paths, key=lambda pos: len(paths[pos]))
    print("farthest", farthest, len(paths[farthest]), paths[farthest])


arena.run_program("ls20", probe)
