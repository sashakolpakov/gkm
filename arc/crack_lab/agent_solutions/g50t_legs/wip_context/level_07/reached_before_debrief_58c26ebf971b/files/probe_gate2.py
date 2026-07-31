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


def metrics(env):
    reward_path, reach = fast_reach(env)
    blobs = connected_components(
        env.frame(), colors=(11, 14, 15), min_area=1)
    auto = next((b.bbox for b in blobs if b.color == 14), None)
    areas = {
        color: sum(b.area for b in blobs if b.color == color)
        for color in (11, 15)
    }
    fronts = tuple((p, len(w))
                   for p, w in _special_frontier(reach, env.frame()))
    return int(env.levels_completed), len(reach), reward_path, auto, areas, fronts


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
    armed = apply(apply(crossed, top_walk + [5]), ARM)
    _, armed_reach = fast_reach(armed)
    bottom, bottom_walk = max(
        _special_frontier(armed_reach, armed.frame()),
        key=lambda item: len(item[1]),
    )
    print("armed", metrics(armed), "bottom", bottom, bottom_walk)

    for padding in range(10):
        macro = [2, 1] * padding + bottom_walk + [5]
        node = apply(armed, macro)
        best = metrics(node)
        best_score = (
            best[0], best[1], -best[4][15], -best[4][11],
            -(best[3][0] if best[3] else 99),
        )
        best_tick = 0
        for tick in range(1, 13):
            node.step(2 if tick % 2 else 1)
            now = metrics(node)
            score = (
                now[0], now[1], -now[4][15], -now[4][11],
                -(now[3][0] if now[3] else 99),
            )
            if score > best_score:
                best, best_score, best_tick = now, score, tick
        print("candidate", padding, macro, "immediate",
              metrics(apply(armed, macro)), "best_tick", best_tick,
              "best", best)


levels, path, err = arena.run_program("g50t", probe)
print("probe_result", levels, len(path), err)
