"""Traverse the observed peg-pair path graphs as ordered dense subgoals."""
import sys
from itertools import product

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
from players import play_level_1
from probe_level2_collect import avatar_key, covered, peg_centers


B_CHAINS = (((4, 2), (2, 0), (0, 5)), ((5, 0), (0, 2), (2, 4)))
C_CHAINS = (((8, 6), (6, 7), (7, 11)), ((11, 7), (7, 6), (6, 8)))


def probe(env):
    play_level_1(env)
    root = env.clone()
    pegs = peg_centers(root.frame())
    for bdir, cdir in product(range(2), repeat=2):
        edges = ((1, 3),) + B_CHAINS[bdir] + C_CHAINS[cdir] + ((9, 10),)
        node = root.clone()
        total = []
        completed = 0
        for edge in edges:
            segment = perception.bounded_bfs(
                node,
                lambda child, _: child.levels_completed > 1
                or (_ and _[-1] == 5 and all(covered(child, pegs[i]) for i in edge)),
                actions=(1, 2, 3, 4, 5),
                key_fn=avatar_key,
                max_states=6000,
                max_depth=40,
            )
            if segment is None:
                break
            for action in segment:
                node.step(action)
                total.append(action)
                if node.levels_completed > 1:
                    print("solved", bdir, cdir, len(total), total)
                    return
            completed += 1
        for action in (5, 5, 5, 5):
            node.step(action)
            total.append(action)
            if node.levels_completed > 1:
                print("solved", bdir, cdir, len(total), total)
                return
        print("trial", bdir, cdir, "edges", completed, "moves", len(total), "level", node.levels_completed)


arena.run_program("cn04", probe)
