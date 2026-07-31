"""Test support subsets before the next gravity reversal in level 7."""

from itertools import combinations
import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import run_actions
from perception import connected_components
from probe_l7_frontier import BASE, R, avatar, summary


def nearest_control(frame):
    position = avatar(frame)
    controls = [
        blob
        for blob in connected_components(frame, colors=(8,), min_area=3)
        if blob.bbox[1] <= 5 and blob.bbox[0] < 63
    ]
    if not controls or position is None:
        return None
    _ax, ay = position
    blob = min(controls, key=lambda item: abs(item.centroid[0] - ay))
    y, x = blob.centroid
    return 6, int(round(x)), int(round(y))


def supports(frame):
    return [
        (int(round(blob.centroid[1])), int(round(blob.centroid[0])))
        for blob in connected_components(frame, colors=(12,), min_area=6)
        if blob.bbox[0] < 63
    ]


def probe(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    run_actions(env, BASE + [R])
    root = env.clone()
    base_level = int(env.levels_completed)
    targets = supports(root.frame())
    print("ROOT", {"targets": targets, **summary(root)})
    survivors = []
    for size in range(len(targets) + 1):
        for chosen in combinations(targets, size):
            node = root.clone()
            run_actions(node, [(6, x, y) for x, y in chosen])
            if node.terminal():
                continue
            action = nearest_control(node.frame())
            if action is None:
                continue
            node.step(*action)
            if node.terminal():
                continue
            state = summary(node)
            survivors.append(
                {
                    "removed": chosen,
                    "control": action,
                    "level_delta": int(node.levels_completed) - base_level,
                    "avatar": state["avatar"],
                    "grid": state["grid"],
                }
            )
    print("SURVIVORS", {"count": len(survivors), "items": survivors})


arena.run_program("bp35", probe)
