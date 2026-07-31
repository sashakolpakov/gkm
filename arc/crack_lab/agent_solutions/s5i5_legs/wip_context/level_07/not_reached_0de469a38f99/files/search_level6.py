import heapq
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr
from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)


CONTROLS = (
    ("A", (6, 12, 48)), ("B", (6, 31, 48)), ("C", (6, 50, 48)),
    ("A<", (6, 9, 57)), ("A>", (6, 15, 57)),
    ("B<", (6, 28, 57)), ("B>", (6, 34, 57)),
    ("C<", (6, 47, 57)), ("C>", (6, 53, 57)),
)
TARGET = {(33, 52), (34, 51), (34, 53), (35, 52)}


def key(env):
    return arr(env.frame())[:45].tobytes()


def marker(frame):
    grid = arr(frame)
    cells = [
        (int(r), int(c)) for r, c in zip(*((grid[:45] == 13).nonzero()))
        if (int(r), int(c)) not in TARGET
    ]
    return cells[0] if cells else (99, 99)


def score(frame):
    r, c = marker(frame)
    return abs(r - 34) + abs(c - 52)


def search(root, max_states=18000, max_depth=45):
    serial = 0
    queue = [(0, score(root.frame()), serial, root.clone(), ())]
    seen = {key(root)}
    best = score(root.frame())
    while queue and len(seen) < max_states:
        depth, _, _, node, path = heapq.heappop(queue)
        if depth >= max_depth:
            continue
        for index, (_, action) in enumerate(CONTROLS):
            child = node.clone()
            child.step(*action)
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            child_path = path + (index,)
            if child.levels_completed > 5:
                return child_path, len(seen), best
            child_score = score(child.frame())
            if child_score < best:
                best = child_score
                print("best", best, marker(child.frame()),
                      [CONTROLS[i][0] for i in child_path])
            serial += 1
            heapq.heappush(
                queue,
                (len(child_path), child_score,
                 serial, child, child_path),
            )
    return None, len(seen), best


def run(env):
    for play in (play_level_1, play_level_2, play_level_3, play_level_4, play_level_5):
        play(env)
    path, states, best = search(env)
    print("states", states, "best", best)
    print("path", None if path is None else [CONTROLS[i][0] for i in path])


arena.run_program("s5i5", run)
