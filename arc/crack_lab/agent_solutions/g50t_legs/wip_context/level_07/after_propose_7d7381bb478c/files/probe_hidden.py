import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _special_frontier, fast_reach
from perception import connected_components


SPECIAL = [2, 2, 3, 5]


def summary(env):
    blobs = connected_components(
        env.frame(), colors=(8, 9, 11, 14, 15), min_area=4)
    marker = next(
        (b.bbox for b in blobs
         if b.color == 9 and b.bbox[0] == 1),
        None,
    )
    autonomous = next((b.bbox for b in blobs if b.color == 14), None)
    strips = tuple((b.bbox, b.area) for b in blobs if b.color == 11)
    fixed = tuple((b.color, b.bbox, b.area) for b in blobs
                  if b.color in (8, 15))
    reward_path, reach = fast_reach(env)
    return (
        int(env.levels_completed), marker, autonomous, strips,
        len(reach), reward_path, fixed,
    )


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

    cases = {
        "base": [],
        "S": SPECIAL,
        "SS": SPECIAL * 2,
        "NS": [5] + SPECIAL,
        "SNS": SPECIAL + [5] + SPECIAL,
    }
    for name, actions in cases.items():
        print(name, len(actions), summary(apply(env, actions)))

    # Reach the same special after every patrol phase, then let the autonomous
    # mover run for one full patrol.  Record only state changes.
    for padding in range(10):
        commit = [2, 1] * padding + SPECIAL
        node = apply(env, commit)
        trace = [summary(node)[:6]]
        for tick in range(1, 13):
            node.step(2 if tick % 2 else 1)
            state = summary(node)[:6]
            if state != trace[-1]:
                trace.append(state)
            if int(node.levels_completed) > 6:
                break
        print("phase", padding, "commit", summary(apply(env, commit))[:6],
              "end", trace[-1], "win", int(node.levels_completed) > 6)

    node = apply(env, [2, 1] * 2 + SPECIAL)
    prior_reach = 22
    for tick in range(1, 21):
        node.step(2 if tick % 2 else 1)
        reward_path, reach = fast_reach(node)
        if len(reach) != prior_reach or reward_path is not None:
            print(
                "growth", tick, len(reach), reward_path,
                "frontier",
                [(pos, len(path)) for pos, path
                 in _special_frontier(reach, node.frame())],
                "state", summary(node)[:6],
            )
            prior_reach = len(reach)
        if reward_path is not None or int(node.levels_completed) > 6:
            break


levels, path, err = arena.run_program("g50t", probe)
print("probe_result", levels, len(path), err)
