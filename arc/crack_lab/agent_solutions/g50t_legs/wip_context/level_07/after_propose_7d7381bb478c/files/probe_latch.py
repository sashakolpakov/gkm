import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _special_frontier, fast_reach
from perception import connected_components


FIRST = [2, 1, 2, 1, 2, 2, 3, 5, 2, 1, 2, 1, 2]
ARM = [2, 1, 2, 1, 2, 2, 3, 5]


def apply(env, actions):
    node = env.clone()
    for action in actions:
        node.step(action)
    return node


def brief(env):
    reward_path, reach = fast_reach(env)
    blobs = connected_components(
        env.frame(), colors=(8, 11, 14, 15), min_area=4)
    return (
        int(env.levels_completed), len(reach), reward_path,
        tuple((p, len(w))
              for p, w in _special_frontier(reach, env.frame())),
        tuple((b.color, b.bbox, b.area) for b in blobs),
    )


def probe(env):
    with open("checkpoint.json") as fh:
        path = json.load(fh)["final_path"]
    for action in path:
        env.step(action)
    crossed = apply(env, FIRST)
    _, reach = fast_reach(crossed)
    _, top_walk = max(
        _special_frontier(reach, crossed.frame()),
        key=lambda item: len(item[1]),
    )
    committed = apply(crossed, top_walk + [5])
    armed = apply(committed, ARM)
    print("armed", brief(armed))
    for delay in range(8):
        actions = []
        for tick in range(delay):
            actions.append(2 if tick % 2 == 0 else 1)
        latched = apply(armed, actions + [5])
        print("latch", delay, actions + [5], brief(latched))
        node = latched
        best = brief(node)
        for tick in range(12):
            node = apply(node, [2 if tick % 2 == 0 else 1])
            state = brief(node)
            if (state[0], state[1]) > (best[0], best[1]):
                best = state
        print("after", delay, best[:4])


levels, path, err = arena.run_program("g50t", probe)
print("probe_result", levels, len(path), err)
