import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _special_frontier, fast_reach
from perception import connected_components


FIRST = [2, 1, 2, 1, 2, 2, 3, 5, 2, 1, 2, 1, 2]
ARM = [2, 1, 2, 1, 2, 2, 3, 5]
REOPEN = [2, 1, 2, 2, 3, 5] + [2, 1] * 5


def apply(env, actions):
    for action in actions:
        env.step(action)
        if env.terminal():
            break


def brief(env):
    reward_path, reach = fast_reach(env)
    auto = [
        b.bbox for b in connected_components(
            env.frame(), colors=(14,), min_area=4)
    ]
    fronts = [(pos, len(path))
              for pos, path in _special_frontier(reach, env.frame())]
    return int(env.levels_completed), len(reach), reward_path, fronts, auto


def commit_remote(env):
    _, reach = fast_reach(env)
    fronts = _special_frontier(reach, env.frame())
    remote = max(fronts, key=lambda item: len(item[1]))
    apply(env, remote[1] + [5])
    return remote[0], remote[1] + [5]


def probe(env):
    with open("checkpoint.json") as fh:
        path = json.load(fh)["final_path"]
    apply(env, path)
    prefix = []

    apply(env, FIRST)
    prefix += FIRST
    print("initial_open", brief(env))
    pos, macro = commit_remote(env)
    prefix += macro
    print("commit", 0, pos, macro, brief(env))

    for cycle in range(1, 9):
        for label, actions in (("arm", ARM), ("reopen", REOPEN)):
            apply(env, actions)
            prefix += actions
            print(label, cycle, brief(env))
            reward_path, _ = fast_reach(env)
            if reward_path is not None:
                apply(env, reward_path)
                prefix += reward_path
                print("win_walk", reward_path, brief(env))
                break
        if int(env.levels_completed) > 6:
            break
        reward_path, _ = fast_reach(env)
        if reward_path is not None:
            apply(env, reward_path)
            prefix += reward_path
            break
        pos, macro = commit_remote(env)
        apply_result = brief(env)
        prefix += macro
        print("commit", cycle, pos, macro, apply_result)
        if int(env.levels_completed) > 6:
            break

    print("result", int(env.levels_completed), len(prefix), prefix)


levels, path, err = arena.run_program("g50t", probe)
print("probe_result", levels, len(path), err)
