"""Measure repeated pressure-pad transfers in two level-7 contexts."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import arr


with open("checkpoint.json") as checkpoint_file:
    PREFIX = json.load(checkpoint_file)["final_path"]


def probe(env):
    for action in PREFIX:
        env.step(*action)

    def surfaces(node):
        frame = arr(node.frame())
        fluid = (frame == 3) | (frame == 11) | (frame == 14) | (frame == 15)

        def top(rows, cols):
            points = fluid[rows, cols].nonzero()[0]
            return None if len(points) == 0 else rows.start + int(points.min())

        return (
            top(slice(8, 30), slice(8, 22)),
            top(slice(8, 56), slice(24, 40)),
            top(slice(8, 30), slice(42, 56)),
            top(slice(30, 56), slice(8, 22)),
            top(slice(30, 56), slice(42, 56)),
        )

    first = env.clone()
    for action in ((6, 24, 8), (6, 42, 8), (6, 40, 19)):
        first.step(*action)
    trace = [surfaces(first)]
    for _ in range(10):
        first.step(6, 24, 8)
        trace.append(surfaces(first))
    print("center_to_upper_left", trace)

    second = env.clone()
    staging = (
        [(6, 24, 8), (6, 42, 8), (6, 40, 19)]
        + [(6, 24, 8)] * 9
        + [(6, 38, 8)] * 4
        + [(6, 24, 32)] * 4
        + [(6, 20, 8)] * 3
        + [(6, 20, 32)] * 3
        + [(6, 22, 41)]
        + [(6, 20, 8)] * 6
        + [(6, 24, 8)] * 4
        + [(6, 38, 32)] * 2
        + [(6, 40, 41)]
        + [(6, 20, 32)] * 6
    )
    for action in staging:
        second.step(*action)
    trace = [surfaces(second)]
    for _ in range(7):
        second.step(6, 38, 32)
        trace.append(surfaces(second))
    print("center_to_lower_right", trace)


arena.run_program("vc33", probe)
