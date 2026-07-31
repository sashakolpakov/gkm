"""Replay the shortest engage-and-bridge route for each level-2 mover."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception
from players import play_level_1
from probe_level2_collect import avatar_key, covered, peg_centers


ROUTES = {
    "A": [2] + [4] * 14,
    "B": [5, 5] + [2] * 4 + [3] + [2] * 3 + [3] * 8,
    "C": [2, 1, 3, 5] + [4] * 9 + [1],
}
SOURCE = {"A": (1, 3), "B": (0, 2, 4, 5), "C": (6, 7, 8, 11)}
SELECT = {"B": 14, "C": 11}


def select_color(node, color):
    ys, xs = np.where(perception.arr(node.frame()) == color)
    action = (6, int(xs[0]), int(ys[0]))
    node.step(*action)
    return action


def probe(env):
    play_level_1(env)
    pegs = peg_centers(env.frame())
    full_path = []
    for name in "ABC":
        if name in SELECT:
            full_path.append(select_color(env, SELECT[name]))
        seen = set()
        for action in ROUTES[name]:
            env.step(action)
            full_path.append(action)
            seen |= {i for i in SOURCE[name] if covered(env, pegs[i])}
            if env.levels_completed > 1:
                print("solved", name, "path", full_path)
                return
        print("stage", name, "source_seen", sorted(seen),
              "level", env.levels_completed)
    full_path.append(select_color(env, 9))
    if env.levels_completed > 1:
        print("solved", "D-select", "path", full_path)
        return
    finish = perception.bounded_bfs(
        env,
        perception.level_goal(1),
        actions=(1, 2, 3, 4, 5),
        key_fn=avatar_key,
        max_states=5000,
        max_depth=40,
    )
    print("D_finish", finish)
    if finish is not None:
        print("solved", "path", full_path + finish)
    else:
        print("unsolved", "path", full_path,
              "colors", perception.color_counts(env.frame()))


arena.run_program("cn04", probe)
