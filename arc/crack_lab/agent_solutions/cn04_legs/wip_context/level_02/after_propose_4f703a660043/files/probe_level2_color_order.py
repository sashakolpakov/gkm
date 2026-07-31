"""Visit colored logical cells in the visible 14 -> 11 -> 9 chain."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
from players import play_level_1
from probe_level2_match import logical_cells


def probe(env):
    play_level_1(env)
    root = env.clone()
    total = []
    for color in (14, 11, 9):
        target = logical_cells(root.frame(), color)
        visited = set()
        while target - visited and len(total) < 92:
            node = perception.replay(root, total)
            segment = perception.bounded_bfs(
                node,
                lambda child, _: child.levels_completed > 1
                or bool(logical_cells(child.frame(), 0) & (target - visited)),
                actions=(1, 2, 3, 4, 5),
                key_fn=lambda child: logical_cells(child.frame(), 0),
                max_states=5000,
                max_depth=35,
            )
            if segment is None:
                break
            total.extend(segment)
            node = perception.replay(root, total)
            newly = logical_cells(node.frame(), 0) & (target - visited)
            visited |= newly
            if node.levels_completed > 1:
                print("solved", len(total), total)
                return
        print("color", color, "covered", len(visited), "/", len(target), "moves", len(total))
    node = perception.replay(root, total)
    print("level", node.levels_completed, "path", total)
    for turns in range(1, 5):
        child = perception.replay(node, [5] * turns)
        print("finish_turns", turns, "level", child.levels_completed)


arena.run_program("cn04", probe)
