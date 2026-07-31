"""Bounded reward search for cn04 level 2 using only observed frames."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
from players import play_level_1


def probe(env):
    play_level_1(env)
    path = perception.bounded_bfs(
        env,
        perception.level_goal(1),
        actions=(1, 2, 3, 4, 5),
        key_fn=lambda node: (
            node.levels_completed,
            (perception.arr(node.frame())[1:] == 0).tobytes(),
        ),
        max_states=10000,
        max_depth=100,
    )
    print("reward_path", path)
    if path is not None:
        print("reward_len", len(path), "result", perception.path_result(env, path))


arena.run_program("cn04", probe)
