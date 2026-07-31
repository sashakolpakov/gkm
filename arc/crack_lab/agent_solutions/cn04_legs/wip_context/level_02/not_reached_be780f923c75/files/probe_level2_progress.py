"""Drive cn04 level 2 by shortest paths to successive meter advances."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
from players import play_level_1


def score(env):
    return perception.color_counts(perception.arr(env.frame())[0]).get(0, 0)


def probe(env):
    play_level_1(env)
    root = env.clone()
    total = []
    for phase in range(40):
        node = perception.replay(root, total)
        if node.levels_completed > 1:
            print("solved", len(total), total)
            return
        before = score(node)
        segment = perception.bounded_bfs(
            node,
            lambda child, _: child.levels_completed > 1 or score(child) > before,
            actions=(1, 2, 3, 4, 5),
            max_states=3500,
            max_depth=28,
        )
        print("phase", phase, "score", before, "segment", segment)
        if segment is None:
            print("stuck", len(total), total)
            return
        total.extend(segment)
    print("limit", len(total), total)


arena.run_program("cn04", probe)
