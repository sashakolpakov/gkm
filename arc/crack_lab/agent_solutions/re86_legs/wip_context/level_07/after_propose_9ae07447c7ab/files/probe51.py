import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import arr
from probe48 import (
    DOWN, LEFT, NAME, PATH, RIGHT, UP, USE,
    bare_frame, descriptor, dot,
)


PLANS = {
    "below_fixture_up": [UP] * 3 + [RIGHT] * 4 + [UP] * 8,
    "left_fixture_right": [UP] * 7 + [RIGHT] * 8,
    "right_fixture_left": [UP] * 3 + [RIGHT] * 9 + [UP] * 7 + [LEFT] * 8,
    "old_square": (
        [UP] * 3 + [RIGHT] * 4 + [UP] * 8
        + [RIGHT] * 9 + [DOWN] * 10
    ),
}


def brief(node, bare, center):
    frame = arr(node.frame())
    desc = descriptor(frame, bare, "outline", center)
    return center, desc[0], desc[2], len(desc[3])


def run(env):
    for action in PATH:
        env.step(action)
    root = env.clone()
    root.step(USE)
    bare = bare_frame(root)
    start = (51, 21)
    print("start", brief(root, bare, start))
    for tag, plan in PLANS.items():
        node = root.clone()
        print(tag)
        last = None
        center = start
        for index, action in enumerate(plan, 1):
            node.step(action)
            dr, dc = {UP: (-3, 0), DOWN: (3, 0),
                      LEFT: (0, -3), RIGHT: (0, 3)}[action]
            center = (center[0] + dr, center[1] + dc)
            state = brief(node, bare, center)
            if state[2:] != (last[2:] if last else None):
                print(index, NAME[action], state)
            last = state


A.run_program("re86", run)
