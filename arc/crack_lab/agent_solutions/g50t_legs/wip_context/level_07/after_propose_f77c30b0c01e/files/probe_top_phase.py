import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _special_frontier, fast_reach
from perception import connected_components


EXPOSED = (
    [2, 1, 2, 1, 2, 2, 3, 5, 2, 1, 2, 1, 2]
    + [1, 3, 3, 1, 1, 5]
    + [2, 1, 2, 1, 2, 2, 3, 5]
    + [2, 1, 2, 2, 3, 5] + [2, 1] * 5
)


def apply(env, actions):
    node = env.clone()
    for action in actions:
        node.step(action)
    return node


def light(env):
    blobs = connected_components(
        env.frame(), colors=(11, 14, 15), min_area=4)
    auto = next((b.bbox for b in blobs if b.color == 14), None)
    areas = tuple(
        (color, sum(b.area for b in blobs if b.color == color))
        for color in (11, 15)
    )
    return auto, areas


def full(env):
    reward_path, reach = fast_reach(env)
    return (
        int(env.levels_completed),
        len(reach),
        reward_path,
        tuple((p, len(w))
              for p, w in _special_frontier(reach, env.frame())),
        light(env),
    )


def probe(env):
    with open("checkpoint.json") as fh:
        checkpoint = json.load(fh)["final_path"]
    for action in checkpoint + EXPOSED:
        env.step(action)
    print("exposed", full(env))
    _, reach = fast_reach(env)
    _, top_walk = max(
        _special_frontier(reach, env.frame()),
        key=lambda item: len(item[1]),
    )

    for padding in range(10):
        macro = [2, 1] * padding + top_walk + [5]
        node = apply(env, macro)
        trace = [light(node)]
        best = full(node)
        best_tick = 0
        for tick in range(1, 25):
            node.step(2 if tick % 2 else 1)
            now_light = light(node)
            if now_light != trace[-1]:
                trace.append(now_light)
                now = full(node)
                if (now[0], now[1], bool(now[3])) > (
                        best[0], best[1], bool(best[3])):
                    best, best_tick = now, tick
        print("phase", padding, "macro", macro,
              "immediate", full(apply(env, macro)),
              "best_tick", best_tick, "best", best,
              "trace", trace)


levels, path, err = arena.run_program("g50t", probe)
print("probe_result", levels, len(path), err)
