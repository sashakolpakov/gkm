import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _special_frontier, fast_reach
from perception import connected_components


PREFIX = (
    [2, 1, 2, 1, 2, 2, 3, 5, 2, 1, 2, 1, 2]
    + [1, 3, 3, 1, 1, 5]
    + [2, 1, 2, 1, 2, 2, 3, 5]
    + [2, 1, 2, 2, 3, 5] + [2, 1] * 5
    + [2, 1] * 5 + [3, 3, 1, 1, 5] + [2]
    + [2, 3, 5] + [2]
    + [2, 2, 3, 5]
)


def brief(env):
    reward_path, reach = fast_reach(env)
    auto = next(
        (b.bbox for b in connected_components(
            env.frame(), colors=(14,), min_area=4)),
        None,
    )
    fronts = tuple((p, len(w))
                   for p, w in _special_frontier(reach, env.frame()))
    return int(env.levels_completed), len(reach), reward_path, fronts, auto


def probe(env):
    with open("checkpoint.json") as fh:
        checkpoint = json.load(fh)["final_path"]
    for action in checkpoint + PREFIX:
        env.step(action)
    last = None
    for tick in range(61):
        state = brief(env)
        if state != last:
            print("tick", tick, state)
            last = state
        if state[0] > 6 or state[2] is not None:
            break
        env.step(2 if tick % 2 == 0 else 1)


levels, path, err = arena.run_program("g50t", probe)
print("probe_result", levels, len(path), err)
