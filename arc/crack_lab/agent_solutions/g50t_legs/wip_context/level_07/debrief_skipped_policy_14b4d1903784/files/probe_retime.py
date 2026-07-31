import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _special_frontier, fast_reach


FIRST = [2, 1, 2, 1, 2, 2, 3, 5, 2, 1, 2, 1, 2]
SPECIAL = [2, 2, 3, 5]


def apply(env, actions):
    node = env.clone()
    for action in actions:
        node.step(action)
    return node


def probe(env):
    with open("checkpoint.json") as fh:
        path = json.load(fh)["final_path"]
    for action in path:
        env.step(action)

    crossed = apply(env, FIRST)
    _, reach = fast_reach(crossed)
    top_pos, top_walk = max(
        _special_frontier(reach, crossed.frame()),
        key=lambda item: len(item[1]),
    )
    committed = apply(crossed, top_walk + [5])
    print("committed", top_pos, top_walk + [5])

    for padding in range(10):
        node = apply(committed, [2, 1] * padding + SPECIAL)
        best = (22, None, 0, ())
        for tick in range(16):
            reward_path, reach = fast_reach(node)
            fronts = tuple((p, len(w))
                           for p, w in _special_frontier(
                               reach, node.frame()))
            candidate = (len(reach), reward_path, tick, fronts)
            if (candidate[1] is not None, candidate[0]) > (
                    best[1] is not None, best[0]):
                best = candidate
            if reward_path is not None or int(node.levels_completed) > 6:
                break
            node.step(2 if tick % 2 == 0 else 1)
        print("padding", padding, "best", best)


levels, path, err = arena.run_program("g50t", probe)
print("probe_result", levels, len(path), err)
