import numpy as np

import gkm_try as harness
from perception import connected_components


def endpoints(frame):
    return sorted((round(b.centroid[0]), round(b.centroid[1]))
                  for b in connected_components(frame, colors=(8,), min_area=9)
                  if b.area == 9)


def probe(env):
    harness.resumed_solve(env)
    for name, selection in (
        ("central", None),
        ("top", (6, 54, 6)),
        ("left", (6, 5, 38)),
        ("right", (6, 47, 47)),
    ):
        for turns in range(9):
            node = env.clone()
            if selection:
                node.step(*selection)
            for _ in range(turns):
                node.step(5)
            print(name, turns, endpoints(node.frame()))


harness.A.run_program("cn04", probe)
