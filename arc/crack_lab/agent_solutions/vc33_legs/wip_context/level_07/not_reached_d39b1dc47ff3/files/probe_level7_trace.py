"""Trace the verified first relay stages with compact symbolic states."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import align_marker_pair_with_pressure_controls
from perception import connected_components


with open("checkpoint.json") as checkpoint_file:
    PREFIX = json.load(checkpoint_file)["final_path"]


class StepLog:
    def __init__(self, env):
        self.env = env
        self.actions = []

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, *action):
        self.actions.append(tuple(action))
        return self.env.step(*action)


def probe(env):
    for action in PREFIX:
        env.step(*action)
    start_level = env.levels_completed

    def state(node):
        markers = {}
        for color in (11, 14, 15):
            found = [
                blob
                for blob in connected_components(
                    node.frame(), colors=(color,), min_area=1
                )
                if blob.area <= 5
            ]
            markers[color] = [
                (blob.area, tuple(round(value, 1) for value in blob.centroid))
                for blob in found
            ]
        gates = [
            (
                blob.color,
                blob.area,
                blob.bbox,
                tuple(round(value, 1) for value in blob.centroid),
            )
            for blob in connected_components(
                node.frame(), colors=(1, 12, 13, 14, 15), min_area=8
            )
        ]
        return {
            "level": node.levels_completed,
            "markers": markers,
            "gates": gates,
        }

    first_cross = ((6, 24, 8), (6, 42, 8), (6, 40, 19))
    for action in first_cross:
        env.step(*action)
    print("cross15", first_cross, state(env))

    logged = StepLog(env)
    align_marker_pair_with_pressure_controls(
        logged,
        marker_color=15,
        max_stages=24,
        max_states=1200,
        max_depth=18,
    )
    print("align15", logged.actions, state(env))

    transfer = (6, 38, 32)
    for _ in range(3):
        env.step(*transfer)
        print("transfer", transfer, state(env))

    affordances = [
        (6, int(blob.centroid[1]), int(blob.centroid[0]))
        for blob in connected_components(
            env.frame(), colors=(9, 12, 13, 14, 15), min_area=2
        )
        if blob.color == 9 or blob.area >= 8
    ]
    print(
        "ready",
        env.levels_completed - start_level,
        affordances,
        state(env),
    )


arena.run_program("vc33", probe)
