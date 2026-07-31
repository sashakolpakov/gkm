"""Check one real replay for every reachable placement of level-3 rare tokens."""
from collections import deque
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from model_l3 import POSITIONS
from search_l3 import ACTIONS, PREFIX


PERMS = (
    (1, 2, 7, 3, 4, 5, 0, 11, 8, 6, 10, 15, 12, 9, 14,
     19, 16, 13, 18, 22, 20, 17, 26, 23, 21, 24, 25, 27, 28, 29),
    (0, 1, 2, 4, 5, 8, 6, 3, 12, 9, 7, 11, 16, 13, 10,
     15, 20, 17, 14, 19, 23, 21, 18, 29, 24, 25, 26, 22, 27, 28),
)


def destinations(permutation):
    result = [None] * len(permutation)
    for destination, source in enumerate(permutation):
        result[source] = destination
    return tuple(result)


MOVES = tuple(destinations(permutation) for permutation in PERMS)


def run(env):
    for action in PREFIX:
        env.step(*action)
    root = env.clone()
    frame = np.asarray(root.frame())
    values = tuple(int(frame[r, c]) for r, c in POSITIONS)
    start = (values.index(11), values.index(12))

    queue = deque([(start, "")])
    seen = {start}
    paths = []
    while queue:
        pair, path = queue.popleft()
        paths.append((pair, path))
        for symbol, move in zip("LR", MOVES):
            child = (move[pair[0]], move[pair[1]])
            if child not in seen:
                seen.add(child)
                queue.append((child, path + symbol))

    print("pair-states", len(paths), "max-depth",
          max(map(lambda item: len(item[1]), paths)), flush=True)
    for checked, (pair, path) in enumerate(paths, 1):
        clone = root.clone()
        for symbol in path:
            clone.step(*ACTIONS[symbol == "R"])
            if clone.levels_completed > 2:
                print("FOUND", path, "pair", pair, "checked", checked,
                      flush=True)
                return
        if checked % 100 == 0:
            print("checked", checked, flush=True)
    print("no-pair-goal", len(paths), flush=True)


if __name__ == "__main__":
    A.run_program("lp85", run)
