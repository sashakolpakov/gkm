import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import arr
from probe48 import DOWN, LEFT, PATH, RIGHT, UP, USE, bare_frame, covered, descriptor, dot


TARGETS = ((18, 57), (24, 39))
ROUTES = {
    "direct-ur": [UP] * 12 + [RIGHT] * 5,
    "direct-ru": [RIGHT] * 5 + [UP] * 12,
    "left-up-right": [LEFT] * 5 + [UP] * 12 + [RIGHT] * 10,
    "left-paint-target": [LEFT] * 5 + [UP] * 14 + [RIGHT] * 3 + [DOWN] * 2 + [RIGHT] * 7,
    "up-left-paint-target": [UP] * 14 + [LEFT] * 2 + [DOWN] * 2 + [RIGHT] * 7,
    "right-up-left": [RIGHT] * 7 + [UP] * 12 + [LEFT] * 2,
}


def run(env):
    for action in PATH:
        env.step(action)
    root = env.clone()
    root.step(USE)
    root.step(USE)
    bare = bare_frame(root)
    for name, route in ROUTES.items():
        node = root.clone()
        for action in route:
            node.step(action)
        frame = arr(node.frame())
        desc = descriptor(frame, bare, "large-cross", dot(frame))
        print(name, desc[:3] if desc else None, covered(desc, "large-cross", TARGETS))


A.run_program("re86", run)
