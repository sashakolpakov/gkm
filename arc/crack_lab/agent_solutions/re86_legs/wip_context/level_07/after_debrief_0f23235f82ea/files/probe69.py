import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import DOWN, LEFT, RIGHT, UP, arr
from probe48 import bare_frame, descriptor


PATH = json.load(open("checkpoint.json"))["final_path"]
TARGETS = ((30, 45), (48, 39), (48, 51))
DEFORM_PLACE = (
    (RIGHT,) * 10 + (UP,) * 7 + (RIGHT,) + (DOWN,) * 4
)
PAINT = (
    (UP,) * 9
    + (LEFT,) * 6
    + (DOWN,) * 2
    + (RIGHT,) * 6
    + (DOWN,) * 7
)
DELTA = {UP: (-3, 0), DOWN: (3, 0), LEFT: (0, -3), RIGHT: (0, 3)}


def run(env):
    for action in PATH:
        env.step(action)
    root = env.clone()
    bare = bare_frame(root)
    center = (48, 12)
    for action in DEFORM_PLACE:
        root.step(action)
        dr, dc = DELTA[action]
        center = center[0] + dr, center[1] + dc
    before = descriptor(arr(root.frame()), bare, "small-cross", center)
    print(
        "PLACED",
        center,
        before[:3],
        tuple(target in before[3] for target in TARGETS),
        flush=True,
    )
    prior_color = before[0]
    for index, action in enumerate(PAINT, 1):
        root.step(action)
        dr, dc = DELTA[action]
        center = center[0] + dr, center[1] + dc
        current = descriptor(
            arr(root.frame()), bare, "small-cross", center
        )
        if current[0] != prior_color:
            print(
                "COLOR",
                index,
                action,
                prior_color,
                current[:3],
                flush=True,
            )
            prior_color = current[0]
    after = descriptor(arr(root.frame()), bare, "small-cross", center)
    print(
        "PAINTED",
        center,
        after[:3],
        tuple(target in after[3] for target in TARGETS),
        "level",
        root.levels_completed,
        flush=True,
    )


A.run_program("re86", run)
