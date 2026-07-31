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
    + [3, 3, 1, 1, 5]
)


def apply(env, actions):
    node = env.clone()
    for action in actions:
        node.step(action)
    return node


def state(env):
    reward_path, reach = fast_reach(env)
    auto = next(
        (b.bbox for b in connected_components(
            env.frame(), colors=(14,), min_area=4)),
        None,
    )
    fronts = tuple((p, len(w))
                   for p, w in _special_frontier(reach, env.frame()))
    return int(env.levels_completed), len(reach), reward_path, auto, fronts


def probe(env):
    with open("checkpoint.json") as fh:
        checkpoint = json.load(fh)["final_path"]
    for action in checkpoint + PREFIX:
        env.step(action)
    print("start", state(env))
    _, reach = fast_reach(env)
    bottom, walk = max(
        _special_frontier(reach, env.frame()),
        key=lambda item: len(item[1]),
    )
    print("bottom", bottom, walk)

    for padding in range(10):
        macro = [2, 1] * padding + walk + [5]
        node = apply(env, macro)
        best = state(node)
        best_tick = 0
        for tick in range(1, 31):
            node.step(2 if tick % 2 else 1)
            now = state(node)
            if (now[0], now[1], bool(now[4])) > (
                    best[0], best[1], bool(best[4])):
                best, best_tick = now, tick
            if now[0] > 6 or now[2] is not None:
                break
        print("candidate", padding, "tick", best_tick, "best", best)


levels, path, err = arena.run_program("g50t", probe)
print("probe_result", levels, len(path), err)
