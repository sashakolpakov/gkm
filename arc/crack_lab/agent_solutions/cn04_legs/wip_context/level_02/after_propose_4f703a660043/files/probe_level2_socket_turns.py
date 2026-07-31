"""Test level-1's three-turn socket maneuver along level-2 peg chains."""
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
    for extra_turns, bdir, cdir in product((1, 2, 3), range(2), range(2)):
        node = root.clone()
        total = []
        edges = ((1, 3),) + B_CHAINS[bdir] + C_CHAINS[cdir] + ((9, 10),)
        for edge in edges:
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
            for action in segment + [5] * extra_turns:
                node.step(action)
                total.append(action)
                if node.levels_completed > 1:
                    print(
                        "solved", extra_turns, bdir, cdir,
                        len(total), total,
                    )
                    return
        print(
            "trial", extra_turns, bdir, cdir,
            "moves", len(total), "level", node.levels_completed,
        )


arena.run_program("cn04", probe)
