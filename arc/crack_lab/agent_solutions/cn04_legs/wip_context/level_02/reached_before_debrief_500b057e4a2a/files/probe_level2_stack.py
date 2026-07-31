"""Stage all selectable figures at the initial avatar position."""
import sys
from itertools import product

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from players import play_level_1


SELECT = {"B": (45, 15), "C": (18, 39), "D": (51, 51)}
MOVE = {
    "B": [1] + [3] * 9,
    "C": [1] * 8 + [3] * 2,
    "D": [1] * 14 + [3] * 13,
}


def probe(env):
    play_level_1(env)
    root = env.clone()
    for turns in product(range(4), repeat=3):
        node = root.clone()
        path = []
        for name, count in zip("BCD", turns):
            action = (6, *SELECT[name])
            node.step(*action)
            path.append(action)
            for move in [5] * count + MOVE[name]:
                node.step(move)
                path.append(move)
                if node.levels_completed > 1:
                    print("solved", turns, len(path), path)
                    return
        for move in (5, 5, 5, 5):
            node.step(move)
            path.append(move)
            if node.levels_completed > 1:
                print("solved", turns, len(path), path)
                return
    print("unsolved", 64)


arena.run_program("cn04", probe)
