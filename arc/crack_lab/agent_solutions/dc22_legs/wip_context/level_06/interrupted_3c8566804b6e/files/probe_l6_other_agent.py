"""Check whether the central inner object moves after the ring lift."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_right import enter_right, movement_reach


UP_CONTROL = (6, 50, 34)


def black_components(env):
    return [
        (blob.bbox, blob.area)
        for blob in perception.connected_components(
            env.frame(), colors=(0,), min_area=4
        )
        if blob.area < 500 and blob.bbox[1] < 32
    ]


def observe(env):
    solve.solve(env)
    node = enter_right(env, 3)
    reached, _ = movement_reach(node)
    for action in reached[(56, 34)]:
        node.step(action)
    print("INNER_BEFORE", black_components(node))
    node.step(*UP_CONTROL)
    print("INNER_AFTER", black_components(node))
    before = perception.arr(node.frame()).copy()
    for action in (1, 2, 3, 4):
        branch = node.clone()
        branch.step(action)
        delta = perception.frame_delta(before, branch.frame())
        central = [
            sample for sample in delta["samples"]
            if sample[0] < 44 and sample[1] < 32
        ]
        print(
            "INNER_ACTION", action, black_components(branch),
            central,
        )


arena.run_program("dc22", observe)
