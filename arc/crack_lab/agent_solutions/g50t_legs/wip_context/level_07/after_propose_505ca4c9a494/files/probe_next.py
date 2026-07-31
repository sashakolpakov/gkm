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
SECOND_REMOTE = [2, 1] * 5 + [3, 3, 1, 1, 5] + [2]


def apply(env, actions):
    node = env.clone()
    for action in actions:
        node.step(action)
    return node


def auto_row(env):
    blob = next(
        (b for b in connected_components(
            env.frame(), colors=(14,), min_area=4)),
        None,
    )
    return None if blob is None else blob.bbox[0]


def full(env):
    reward_path, reach = fast_reach(env)
    return (
        int(env.levels_completed),
        len(reach),
        reward_path,
        tuple((p, len(w))
              for p, w in _special_frontier(reach, env.frame())),
        auto_row(env),
    )


def probe(env):
    with open("checkpoint.json") as fh:
        checkpoint = json.load(fh)["final_path"]
    for action in checkpoint + EXPOSED + SECOND_REMOTE:
        env.step(action)
    print("start", full(env))
    _, reach = fast_reach(env)

    for special, walk in _special_frontier(reach, env.frame()):
        for padding in range(10):
            macro = [2, 1] * padding + walk + [5]
            node = apply(env, macro)
            best_row = auto_row(node)
            best_tick = 0
            win_tick = None
            for tick in range(1, 31):
                node.step(2 if tick % 2 else 1)
                row = auto_row(node)
                if row is not None and (best_row is None or row < best_row):
                    best_row, best_tick = row, tick
                if int(node.levels_completed) > 6:
                    win_tick = tick
                    break
            print("candidate", special, padding,
                  "immediate_row", auto_row(apply(env, macro)),
                  "min", best_row, "tick", best_tick,
                  "win", win_tick, "end", full(node))


levels, path, err = arena.run_program("g50t", probe)
print("probe_result", levels, len(path), err)
