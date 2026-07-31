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


def compact(env):
    reward_path, reach = fast_reach(env)
    blobs = connected_components(
        env.frame(), colors=(8, 11, 14, 15), min_area=4)
    return (
        int(env.levels_completed), len(reach), reward_path,
        tuple((b.color, b.bbox, b.area) for b in blobs),
        tuple((p, len(w))
              for p, w in _special_frontier(reach, env.frame())),
    )


def run(label, node, ticks=40):
    last = None
    for tick in range(ticks + 1):
        state = compact(node)
        # Report mover/barrier changes compactly, but suppress timer-only noise.
        key = state[:2] + state[3:]
        if key != last:
            print(label, tick, state)
            last = key
        if state[0] > 6 or state[2] is not None:
            return
        node.step(2 if tick % 2 == 0 else 1)


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
    run("direct", apply(committed, ARM))

    failed = apply(committed, FIRST)
    _, failed_reach = fast_reach(failed)
    bottom, bottom_walk = max(
        _special_frontier(failed_reach, failed.frame()),
        key=lambda item: len(item[1]),
    )
    print("bottom", bottom, bottom_walk + [5])
    recommitted = apply(failed, bottom_walk + [5])
    run("recommit", apply(recommitted, ARM))


levels, path, err = arena.run_program("g50t", probe)
print("probe_result", levels, len(path), err)
