import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import perception as P
from probe_level6_lower import brief, click, pieces, stage_upper


def probe(env):
    stage_upper(env)
    env.step(2)
    env.step(2)
    groups = pieces(env.frame())
    targets = {
        "carrier": next(iter(groups[1])),
        "bridge": next(iter(groups[2])),
        "peg": next(iter(groups[3])),
    }
    print("root", brief(env))
    for label, target in targets.items():
        selected = env.clone()
        before = selected.frame()
        click(selected, target)
        delta = P.frame_delta(before, selected.frame())
        print("selected", label, target, delta["count"], delta["bbox"])
        for action in (1, 2, 3, 4):
            node = selected.clone()
            trace = []
            previous = None
            for count in range(5):
                current = brief(node)
                if count == 0 or current != previous:
                    trace.append((count, current))
                previous = current
                node.step(action)
            print(label, action, trace)


A.run_program("lf52", probe)
