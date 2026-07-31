import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import perception as P
import players


def objects(node):
    return [
        (o["color"], o["bbox"], o["area"])
        for o in P.object_candidates(node.frame(), min_area=2)
        if o["color"] in (1, 9, 10, 11, 12, 14)
    ]


def run(env):
    for level in range(1, 6):
        getattr(players, f"play_level_{level}")(env)
    node = env.clone()
    route = (1, 1, (6, 31, 43), 4, 4, 4) + (1,) * 8
    for index, action in enumerate(route, 1):
        if isinstance(action, tuple):
            node.step(*action)
        else:
            node.step(action)
        print("COOP", index, action, node.levels_completed, objects(node))
        if node.terminal():
            break


A.run_program("m0r0", run)
