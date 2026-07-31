"""Track the two visible coordinate affordances through their cycles."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve


TOP_CONTROL = (54, 6)
MAIN_CONTROL = (50, 26)
TRACKED = (0, 1, 2, 7, 8, 9, 10, 11, 12, 14, 15)


def compact_state(env):
    components = perception.connected_components(
        env.frame(), colors=TRACKED, min_area=1
    )
    return [
        (blob.color, blob.bbox, blob.area)
        for blob in components
        if blob.bbox[1] < 40 and blob.area < 1000
    ]


def cycle(root, label, point, repetitions):
    node = root.clone()
    print(label, 0, compact_state(node))
    for index in range(1, repetitions + 1):
        before = perception.arr(node.frame()).copy()
        node.step(6, *point)
        delta = perception.frame_delta(before, node.frame())
        print(label, index, delta["count"], compact_state(node))


def observe(env):
    solve.solve(env)
    cycle(env, "TOP", TOP_CONTROL, 8)
    cycle(env, "MAIN", MAIN_CONTROL, 12)


arena.run_program("dc22", observe)
