"""Cover cn04 level-2 peg groups in the observed color-chain order."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
from players import play_level_1
from probe_level2_collect import avatar_key, covered, peg_centers


GROUPS = (
    (1, 3),          # controlled black piece
    (0, 2, 4, 5),    # color 14
    (6, 7, 8, 11),   # color 11
    (9, 10),         # color 9
)


def probe(env):
    play_level_1(env)
    root = env.clone()
    pegs = peg_centers(root.frame())
    total = []
    for group_number, group in enumerate(GROUPS):
        remaining = set(group)
        while remaining:
            node = perception.replay(root, total)
            segment = perception.bounded_bfs(
                node,
                lambda child, _: child.levels_completed > 1
                or any(covered(child, pegs[i]) for i in remaining),
                actions=(1, 2, 3, 4, 5),
                key_fn=avatar_key,
                max_states=8000,
                max_depth=40,
            )
            if segment is None:
                print("stuck", group_number, sorted(remaining), len(total), total)
                return
            total.extend(segment)
            node = perception.replay(root, total)
            hit = {i for i in remaining if covered(node, pegs[i])}
            remaining -= hit
            print(
                "group",
                group_number,
                "segment",
                segment,
                "hit",
                [(i, pegs[i]) for i in sorted(hit)],
                "moves",
                len(total),
                "level",
                node.levels_completed,
            )
            if node.levels_completed > 1:
                print("solved", len(total), total)
                return
    node = perception.replay(root, total)
    finish = perception.bounded_bfs(
        node,
        perception.level_goal(1),
        actions=(1, 2, 3, 4, 5),
        key_fn=avatar_key,
        max_states=16000,
        max_depth=max(0, 94 - len(total)),
    )
    print("finish", finish)
    if finish is not None:
        total.extend(finish)
        print("solved", len(total), total)


arena.run_program("cn04", probe)
