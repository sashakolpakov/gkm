"""Search reachable level-2 contexts where action 6 has an observable effect."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
from players import play_level_1


def use_effect(node):
    child = node.clone()
    child.step(6)
    delta = perception.frame_delta(
        perception.arr(node.frame())[1:],
        perception.arr(child.frame())[1:],
    )
    return child.levels_completed > node.levels_completed or delta["count"] > 0


def probe(env):
    play_level_1(env)
    path = perception.bounded_bfs(
        env,
        lambda node, _: use_effect(node),
        actions=(1, 2, 3, 4, 5),
        key_fn=lambda node: perception.arr(node.frame())[1:].tobytes(),
        max_states=12000,
        max_depth=80,
    )
    print("use_context_path", path)
    if path is not None:
        node = perception.replay(env, path)
        child = node.clone()
        child.step(6)
        print(
            "before",
            perception.color_counts(node.frame()),
            "after",
            perception.color_counts(child.frame()),
            "delta",
            perception.frame_delta(node.frame(), child.frame()),
            "level",
            child.levels_completed,
        )


arena.run_program("cn04", probe)
