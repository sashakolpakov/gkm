"""Compact symbolic observations for vc33 level 7."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import color_counts, connected_components, frame_delta


with open("checkpoint.json") as checkpoint_file:
    PREFIX = json.load(checkpoint_file)["final_path"]


def probe(env):
    for action in PREFIX:
        env.step(*action)
    start_level = env.levels_completed

    def blobs(node, colors, min_area=1, max_area=None):
        found = connected_components(
            node.frame(), colors=colors, min_area=min_area
        )
        if max_area is not None:
            found = [blob for blob in found if blob.area <= max_area]
        return [
            (
                blob.color,
                blob.area,
                blob.bbox,
                tuple(round(value, 1) for value in blob.centroid),
            )
            for blob in found
        ]

    def state(node):
        return {
            "level": node.levels_completed,
            "markers": blobs(node, (11, 14, 15), max_area=5),
            "fluids": blobs(node, (3, 4, 5, 7), min_area=8),
            "gates": blobs(node, (1, 12, 13, 14, 15), min_area=8),
        }

    controls = [
        (6, int(blob.centroid[1]), int(blob.centroid[0]))
        for blob in connected_components(
            env.frame(), colors=(9,), min_area=2
        )
    ]
    gates = [
        (6, int(blob.centroid[1]), int(blob.centroid[0]))
        for blob in connected_components(
            env.frame(), colors=(12, 13, 14, 15), min_area=8
        )
    ]
    initial = state(env)
    print("initial", env.actions, color_counts(env.frame()), initial)
    print("affordances", {"controls": controls, "active_gates": gates})

    base_frame = env.frame()
    for action in controls + gates:
        child = env.clone()
        child.step(*action)
        print(
            "one",
            action,
            frame_delta(base_frame, child.frame()),
            state(child),
        )

    child = env.clone()
    prefix = ((6, 24, 8), (6, 42, 8), (6, 40, 19))
    for action in prefix:
        child.step(*action)
        print("prefix", action, state(child))
    print("reward_delta", child.levels_completed - start_level)


arena.run_program("vc33", probe)
