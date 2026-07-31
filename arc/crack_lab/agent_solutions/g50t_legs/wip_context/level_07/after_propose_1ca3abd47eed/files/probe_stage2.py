import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _special_frontier, fast_reach
from perception import connected_components


FIRST = [2, 1, 2, 1, 2, 2, 3, 5, 2, 1, 2, 1, 2]


def brief(env):
    reward_path, reach = fast_reach(env)
    blobs = connected_components(
        env.frame(), colors=(9, 11, 14, 15), min_area=4)
    movers = tuple((b.color, b.bbox, b.area) for b in blobs
                   if b.color in (11, 14, 15))
    return (
        int(env.levels_completed), len(reach), reward_path,
        tuple((pos, len(path))
              for pos, path in _special_frontier(reach, env.frame())),
        movers,
    )


def probe(env):
    with open("checkpoint.json") as fh:
        path = json.load(fh)["final_path"]
    for action in path + FIRST:
        env.step(action)
    print("start", brief(env))

    _, reach = fast_reach(env)
    for special, walk in _special_frontier(reach, env.frame()):
        for padding in range(10):
            node = env.clone()
            actions = [1, 2] * padding + walk + [5]
            for action in actions:
                node.step(action)
            best = brief(node)
            best_tick = 0
            for tick in range(1, 16):
                node.step(2 if tick % 2 else 1)
                state = brief(node)
                if (state[0], state[1]) > (best[0], best[1]):
                    best, best_tick = state, tick
                if state[0] > 6:
                    break
            print("candidate", special, padding, len(actions),
                  "best_tick", best_tick, "best", best[:4])


levels, path, err = arena.run_program("g50t", probe)
print("probe_result", levels, len(path), err)
