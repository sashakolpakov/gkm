import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import arr
from probe48 import DOWN, LEFT, NAME, PATH, RIGHT, UP, USE, bare_frame, descriptor


CHAR = {"U": UP, "D": DOWN, "L": LEFT, "R": RIGHT}
BASE = tuple(CHAR[ch] for ch in "ULLUUUUUUUULLLUUURRRDDDDUUUURUUUDDDRRDRLULLLUDLL")
SHIFT = tuple(CHAR[ch] for ch in "RRRRRDRRLLULLLL")
DELTA = {UP: (-3, 0), DOWN: (3, 0), LEFT: (0, -3), RIGHT: (0, 3)}


def run(env):
    for action in PATH:
        env.step(action)
    root = env.clone()
    root.step(USE)
    root.step(USE)
    bare = bare_frame(root)
    center = (54, 24)
    for action in BASE:
        root.step(action)
        dr, dc = DELTA[action]
        center = center[0] + dr, center[1] + dc
    for repeat in range(2):
        print("REPEAT", repeat, center, descriptor(arr(root.frame()), bare, "large-cross", center)[:3])
        for index, action in enumerate(SHIFT, 1):
            root.step(action)
            dr, dc = DELTA[action]
            center = center[0] + dr, center[1] + dc
            desc = descriptor(arr(root.frame()), bare, "large-cross", center)
            print(index, NAME[action], center, desc[:3])
    for action in [UP] * 4 + [DOWN] * 4:
        root.step(action)
        dr, dc = DELTA[action]
        center = center[0] + dr, center[1] + dc
    print("PAINTED", center, descriptor(arr(root.frame()), bare, "large-cross", center)[:3])


A.run_program("re86", run)
