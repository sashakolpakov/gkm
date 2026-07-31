"""Compact observational probes for cn04 level 2."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
from players import play_level_1


def summarize(env):
    objects = [
        (o["color"], o["bbox"], o["area"])
        for o in perception.object_candidates(env.frame(), min_area=4)
    ]
    deltas = {
        action: (delta["count"], delta["bbox"])
        for action, delta in perception.action_deltas(env, env.actions).items()
    }
    print("level", env.levels_completed + 1, "actions", env.actions)
    print("colors", perception.color_counts(env.frame()))
    print("objects", objects)
    print("deltas", deltas)
    for action in env.actions:
        clone = env.clone()
        clone.step(action)
        moving = [
            (b.color, b.bbox, b.area)
            for b in perception.connected_components(
                clone.frame(), colors=(0, 9, 11, 14), min_area=4
            )
        ]
        print("after", action, moving)


def probe(env):
    known = [2] * 7 + [4] * 4 + [5] * 3
    for length in (0, 7, 11, 12, 13, 14):
        state = perception.replay(env, known[:length])
        pieces = [
            (b.color, b.bbox, b.area)
            for b in perception.connected_components(
                state.frame(), colors=(0, 9, 11, 14), min_area=4
            )
        ]
        print("level1_prefix", length, state.levels_completed, pieces)
    play_level_1(env)
    summarize(env)
    path = perception.bounded_bfs(
        env,
        perception.level_goal(1),
        actions=env.actions,
        max_states=3000,
        max_depth=40,
    )
    print("reward_path", path)
    if path is not None:
        print("reward_result", perception.path_result(env, path)["levels_completed"])


arena.run_program("cn04", probe)
