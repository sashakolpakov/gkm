"""Measure each level-6 direction glyph in its own fresh Arena."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr


ROWS = (33, 36, 39, 42, 45, 48)
COLS = (11, 16, 21)
with open("checkpoint.json") as stream:
    CHECKPOINT = json.load(stream)


def run(selector):
    observation = {}

    def observe(env):
        for action in CHECKPOINT["final_path"]:
            env.step(action)
        env.step(6, selector, 58)
        frame = arr(env.frame())
        observation.update(
            {
                "selector": selector,
                "masks": tuple(
                    tuple(index for index, row in enumerate(ROWS) if int(frame[row][col]) == 5)
                    for col in COLS
                ),
            }
        )

    A.run_program("tn36", observe)
    return observation


for value in (5, 15, 25, 35):
    print("fresh_protocol", run(value), flush=True)
