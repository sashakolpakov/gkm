"""Bounded replay BFS for a true 8-9 tether prefix from the strong frontier."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception as p
import players
from search_level5_threaded import (
    CONTROLLED_PAIR,
    LOWER_FINAL_EIGHT,
    PREFIX,
    SUFFIX,
    SURPLUS_TRANSFER,
    apply,
    observe,
    target_prefix,
)


def run(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)
    apply(
        env,
        PREFIX
        + CONTROLLED_PAIR
        + LOWER_FINAL_EIGHT
        + SUFFIX
        + SURPLUS_TRANSFER,
    )
    path = p.bounded_replay_bfs(
        env,
        goal_fn=lambda node, _path: (
            node.levels_completed > 4
            or target_prefix(observe(node)[3]) >= 2
        ),
        action_fn=lambda _node: (1, 2, 3, 4),
        max_states=5000,
        max_depth=45,
    )
    print("PAIR_PATH", path)
    if path is not None:
        node = p.replay(env, path)
        observation = observe(node)
        print(
            "PAIR_STATE",
            node.levels_completed,
            observation[2:5],
            observation[1],
        )


levels, path, error = arena.run_program("sk48", run)
print("END", levels, len(path), error)
