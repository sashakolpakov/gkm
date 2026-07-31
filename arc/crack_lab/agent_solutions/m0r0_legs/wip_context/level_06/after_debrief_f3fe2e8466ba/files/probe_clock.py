import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import perception as P
import players


def summarize(node):
    counts = P.color_counts(node.frame())
    special = [
        (o["color"], o["bbox"], o["area"])
        for o in P.object_candidates(node.frame(), min_area=4)
        if o["color"] in (0, 9, 10, 12, 14)
    ]
    return counts.get(0, 0), special


def run(env):
    for level in range(1, 6):
        getattr(players, f"play_level_{level}")(env)
    for name, prefix in (("hold", (1, 1)), ("leave", (1, 1, 2))):
        node = P.replay(env, prefix)
        print("CLOCK", name, 0, node.levels_completed, summarize(node))
        for turn in range(1, 241):
            node.step(5)
            if turn % 20 == 0 or node.levels_completed > 5:
                print("CLOCK", name, turn, node.levels_completed, summarize(node))
            if node.levels_completed > 5 or node.terminal():
                break


A.run_program("m0r0", run)
