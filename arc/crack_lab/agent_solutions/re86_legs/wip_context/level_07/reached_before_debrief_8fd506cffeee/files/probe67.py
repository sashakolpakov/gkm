import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import DOWN, LEFT, RIGHT, UP, USE, arr
from probe48 import bare_frame, visible_shape


PATH = json.load(open("checkpoint.json"))["final_path"]
TARGETS = ((18, 57), (24, 39))
DEFORM_PLACE = (
    (UP,) * 3
    + (RIGHT,) * 4
    + (UP,) * 2
    + (RIGHT,) * 5
    + (UP,) * 6
)
PAINT = (LEFT,) * 13 + (UP,) * 5 + (DOWN,) * 5 + (RIGHT,) * 13
DELTA = {UP: (-3, 0), DOWN: (3, 0), LEFT: (0, -3), RIGHT: (0, 3)}


def run(env):
    for action in PATH:
        env.step(action)
    root = env.clone()
    root.step(USE)
    bare = bare_frame(root)
    center = (51, 21)
    for action in DEFORM_PLACE:
        root.step(action)
        dr, dc = DELTA[action]
        center = center[0] + dr, center[1] + dc
    before = visible_shape(arr(root.frame()), bare, center)
    print(
        "PLACED",
        center,
        before[0:2],
        (
            min(row for row, _ in before[2]),
            min(col for _, col in before[2]),
            max(row for row, _ in before[2]),
            max(col for _, col in before[2]),
        ),
        tuple(target in before[2] for target in TARGETS),
        flush=True,
    )
    for action in PAINT:
        root.step(action)
        dr, dc = DELTA[action]
        center = center[0] + dr, center[1] + dc
    after = visible_shape(arr(root.frame()), bare, center)
    print(
        "PAINTED",
        center,
        after[0:2],
        len(after[2]),
        tuple(target in after[2] for target in TARGETS),
        "level",
        root.levels_completed,
        flush=True,
    )


A.run_program("re86", run)
