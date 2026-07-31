import json
import sys
from itertools import product

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import arr
from probe48 import (
    DOWN, LEFT, RIGHT, UP, PATH, SPECS, bare_frame, covered, descriptor,
)

BASE = (
    [UP] * 10 + [RIGHT, UP, DOWN, DOWN] + [RIGHT] * 7
    + [UP, RIGHT, RIGHT, RIGHT, RIGHT, UP, DOWN]
)


def run(env):
    for action in PATH:
        env.step(action)
    root = env.clone()
    bare = bare_frame(root)
    kind, wanted, targets = SPECS[0]
    print("base-len", len(BASE))
    for ups in range(1, 5):
        for lefts in range(5, 11):
            node = root.clone()
            route = BASE + [UP] * ups + [LEFT] * lefts + [DOWN] * ups + [RIGHT] * lefts
            for action in route:
                node.step(action)
            desc = descriptor(arr(node.frame()), bare, kind)
            hits = covered(desc, kind, targets)
            if desc and (desc[0] == wanted or len(hits) == len(targets)):
                print("loop", ups, lefts, "len", len(route),
                      "color", desc[0], "hits", len(hits), "desc", desc[:3])
    common = root.clone()
    route = BASE + [UP, LEFT, LEFT, LEFT, LEFT, LEFT, LEFT, LEFT,
                    DOWN, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT]
    for action in route:
        common.step(action)
    for length in range(1, 7):
        for suffix in product((UP, DOWN, LEFT, RIGHT), repeat=length):
            node = common.clone()
            for action in suffix:
                node.step(action)
            desc = descriptor(arr(node.frame()), bare, kind)
            hits = covered(desc, kind, targets)
            if desc and desc[0] == wanted and len(hits) == len(targets):
                print("FIXED", len(route) + length, suffix, desc[:3])
                return
    print("no short fix")


A.run_program("re86", run)
