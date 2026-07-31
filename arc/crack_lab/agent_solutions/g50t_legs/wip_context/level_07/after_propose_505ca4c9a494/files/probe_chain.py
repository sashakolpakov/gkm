import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _special_frontier, fast_reach
from perception import connected_components


OPEN_AND_CROSS = [2, 1, 2, 1, 2, 2, 3, 5, 2, 1, 2, 1, 2]


def state(env):
    reward_path, reach = fast_reach(env)
    fronts = _special_frontier(reach, env.frame())
    blocks = tuple(
        (b.color, b.bbox, b.area)
        for b in connected_components(
            env.frame(), colors=(8, 11, 14, 15), min_area=4)
    )
    return reward_path, reach, fronts, blocks


def apply(env, actions):
    for action in actions:
        env.step(action)
        if env.terminal():
            break


def probe(env):
    with open("checkpoint.json") as fh:
        path = json.load(fh)["final_path"]
    apply(env, path)
    prefix = []
    used = set()
    for cycle in range(1, 8):
        apply(env, OPEN_AND_CROSS)
        prefix += OPEN_AND_CROSS
        reward_path, reach, fronts, blocks = state(env)
        print("opened", cycle, int(env.levels_completed), len(reach),
              reward_path, [(p, len(w)) for p, w in fronts], blocks)
        if reward_path is not None:
            apply(env, reward_path)
            prefix += reward_path
            break
        fresh = [(p, walk) for p, walk in fronts if p not in used]
        if not fresh:
            break
        pos, walk = max(fresh, key=lambda item: len(item[1]))
        macro = walk + [5]
        apply(env, macro)
        prefix += macro
        used.add(pos)
        print("commit", pos, macro, "level", int(env.levels_completed))
        if int(env.levels_completed) > 6:
            break
    print("result", int(env.levels_completed), len(prefix), prefix)


levels, path, err = arena.run_program("g50t", probe)
print("probe_result", levels, len(path), err)
