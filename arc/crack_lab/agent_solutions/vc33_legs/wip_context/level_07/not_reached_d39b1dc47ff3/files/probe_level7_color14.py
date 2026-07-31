"""Test the opposing-platform leg for color 14 after verified staging."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import cross_horizontal_gates_then_align_opposing_markers
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

    def markers():
        result = {}
        for color in (11, 14, 15):
            result[color] = [
                (
                    blob.area,
                    tuple(round(value, 1) for value in blob.centroid),
                )
                for blob in connected_components(
                    env.frame(), colors=(color,), min_area=1
                )
                if blob.area <= 5
            ]
        return result

    staging = [
        (6, 24, 8),
        (6, 42, 8),
        (6, 40, 19),
        *([(6, 24, 8)] * 9),
        *([(6, 38, 8)] * 4),
        *([(6, 24, 32)] * 4),
        *([(6, 20, 8)] * 3),
        *([(6, 20, 32)] * 3),
        (6, 22, 41),
        *([(6, 20, 8)] * 6),
    ]
    for action in staging:
        env.step(*action)
    print("staged", env.levels_completed, markers())

    logged = StepLog(env)
    cross_horizontal_gates_then_align_opposing_markers(
        logged,
        marker_colors=(14,),
        max_stages=32,
        max_states=1800,
        max_depth=24,
    )
    print(
        "color14",
        logged.actions,
        env.levels_completed - start_level,
        markers(),
    )


arena.run_program("vc33", probe)
