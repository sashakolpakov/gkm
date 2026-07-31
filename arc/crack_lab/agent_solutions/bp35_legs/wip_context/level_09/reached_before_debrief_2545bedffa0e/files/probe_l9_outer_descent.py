"""Descend the protected x=3 catch outside the main chamber wall."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, enter_control_row


def probe(env):
    enter_control_row(env)
    env.step(6, 9, 3)
    print("OUTER", compact(env))
    for depth in range(1, 41):
        env.step(6, 3, 33)
        goals = [
            (blob.bbox, blob.area)
            for blob in connected_components(env.frame(), colors=(7,), min_area=3)
            if blob.bbox[0] < 63
        ]
        print("DESCEND", depth, compact(env), "goals", goals)
        if env.terminal() or int(env.levels_completed) >= 9:
            return


arena.run_program("bp35", probe)
