"""Try one three-turn socket maneuver per visible region."""
import sys
from itertools import product

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
from players import play_level_1
from probe_level2_collect import avatar_key, covered, peg_centers


B_EDGES = ((0, 2), (0, 5), (2, 4))
C_EDGES = ((6, 7), (6, 8), (7, 11))


def probe(env):
    play_level_1(env)
    root = env.clone()
    pegs = peg_centers(root.frame())
    for b_edge, c_edge in product(B_EDGES, C_EDGES):
        node = root.clone()
        total = []
        for edge in ((1, 3), b_edge, c_edge, (9, 10)):
            segment = perception.bounded_bfs(
                node,
                lambda child, path: child.levels_completed > 1
                or (path and path[-1] == 5
                    and all(covered(child, pegs[i]) for i in edge)),
                actions=(1, 2, 3, 4, 5),
                key_fn=avatar_key,
                max_states=1800,
                max_depth=45,
            )
            if segment is None:
                break
            for action in segment + [5, 5]:
                node.step(action)
                total.append(action)
                if node.levels_completed > 1:
                    print("solved", b_edge, c_edge, len(total), total)
                    return
        print("trial", b_edge, c_edge, len(total), node.levels_completed)


arena.run_program("cn04", probe)
