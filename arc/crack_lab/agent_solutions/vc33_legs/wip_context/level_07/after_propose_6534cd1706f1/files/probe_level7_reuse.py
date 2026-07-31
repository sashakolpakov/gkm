"""Test existing gate and alignment legs after staging color 15."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import (
    align_marker_pair_with_pressure_controls,
    cross_pressure_gates_then_align_height,
)
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

    def state():
        markers = {}
        for color in (11, 14, 15):
            markers[color] = [
                (
                    blob.area,
                    tuple(round(value, 1) for value in blob.centroid),
                )
                for blob in connected_components(
                    env.frame(), colors=(color,), min_area=1
                )
                if blob.area <= 5
            ]
        gates = [
            (blob.color, blob.bbox)
            for blob in connected_components(
                env.frame(), colors=(1, 12, 13, 14, 15), min_area=8
            )
        ]
        return env.levels_completed, markers, gates

    staging = [
        (6, 24, 8),
        (6, 42, 8),
        (6, 40, 19),
        *([(6, 24, 8)] * 9),
        *([(6, 38, 8)] * 4),
        *([(6, 24, 32)] * 4),
    ]
    for action in staging:
        env.step(*action)
    print("staged", state())

    crossing = StepLog(env)
    cross_pressure_gates_then_align_height(
        crossing,
        marker_color=11,
        max_stages=16,
        max_states=1000,
        max_depth=20,
    )
    print("cross11", crossing.actions, state())

    alignment = StepLog(env)
    align_marker_pair_with_pressure_controls(
        alignment,
        marker_color=11,
        max_stages=24,
        max_states=1000,
        max_depth=18,
    )
    print(
        "align11",
        alignment.actions,
        env.levels_completed - start_level,
        state(),
    )


arena.run_program("vc33", probe)
