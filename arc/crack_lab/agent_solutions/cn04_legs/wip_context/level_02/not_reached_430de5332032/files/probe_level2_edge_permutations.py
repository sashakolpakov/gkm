"""Test all compact orders of the observed peg-pair socket edges."""
import sys
from itertools import permutations

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
    cache = {}

    def route(node, edge):
        key = (avatar_key(node), edge)
        if key not in cache:
            cache[key] = perception.bounded_bfs(
                node,
                lambda child, path: child.levels_completed > 1
                or (path and path[-1] == 5 and all(covered(child, pegs[i]) for i in edge)),
                actions=(1, 2, 3, 4, 5),
                key_fn=avatar_key,
                max_states=2500,
                max_depth=40,
            )
        return cache[key]

    trials = 0
    for b_order in permutations(B_EDGES):
        for c_order in permutations(C_EDGES):
            trials += 1
            node = root.clone()
            total = []
            complete = True
            for edge in ((1, 3),) + b_order + c_order + ((9, 10),):
                segment = route(node, edge)
                if segment is None:
                    complete = False
                    break
                for action in segment:
                    node.step(action)
                    total.append(action)
                    if node.levels_completed > 1:
                        print("solved", len(total), total, "orders", b_order, c_order)
                        return
            if complete:
                for action in (5, 5, 5, 5):
                    node.step(action)
                    total.append(action)
                    if node.levels_completed > 1:
                        print("solved", len(total), total, "orders", b_order, c_order)
                        return
    print("unsolved trials", trials, "cached_routes", len(cache))


arena.run_program("cn04", probe)
