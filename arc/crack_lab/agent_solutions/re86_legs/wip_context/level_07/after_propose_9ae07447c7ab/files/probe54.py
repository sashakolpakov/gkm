import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import arr
from probe48 import DOWN, LEFT, PATH, RIGHT, SPECS, UP, bare_frame, covered, descriptor, dot


def replay(root, actions):
    node = root.clone()
    for action in actions:
        node.step(action)
    return node


def run(env):
    for action in PATH:
        env.step(action)
    bare = bare_frame(env)
    kind, wanted, targets = SPECS[0]
    for middle_rights in range(6, 10):
        base = tuple(
            [UP] * 10
            + [RIGHT, UP, DOWN, DOWN]
            + [RIGHT] * middle_rights
            + [UP]
            + [RIGHT] * 4
            + [UP, DOWN]
        )
        base_node = replay(env, base)
        base_desc = descriptor(
            arr(base_node.frame()), bare, kind, dot(arr(base_node.frame()))
        )
        print(
            "BASE",
            middle_rights,
            base_desc[:3],
            len(covered(base_desc, kind, targets)),
        )
        for vertical in range(1, 6):
            for horizontal in range(5, 12):
                loops = (
                    [LEFT] * horizontal
                    + [UP] * vertical
                    + [DOWN] * vertical
                    + [RIGHT] * horizontal,
                    [UP] * vertical
                    + [LEFT] * horizontal
                    + [RIGHT] * horizontal
                    + [DOWN] * vertical,
                )
                for order, loop in enumerate(loops):
                    node = replay(base_node, loop)
                    frame = arr(node.frame())
                    desc = descriptor(frame, bare, kind, dot(frame))
                    hits = covered(desc, kind, targets)
                    if desc and desc[0] == wanted and len(hits) == len(targets):
                        print("SOLVED", middle_rights, vertical, horizontal, order, base + tuple(loop), desc[:3])
                        return
    print("NO_SIMPLE_LOOP")


A.run_program("re86", run)
