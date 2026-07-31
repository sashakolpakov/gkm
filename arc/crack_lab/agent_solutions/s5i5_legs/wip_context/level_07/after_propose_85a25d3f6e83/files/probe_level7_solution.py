import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, frame_delta


BUTTONS = {
    "A<": (6, 4, 51), "A>": (6, 10, 51), "B": (6, 17, 51),
    "C<": (6, 26, 51), "C>": (6, 32, 51), "D": (6, 39, 51),
    "F<": (6, 4, 58), "F>": (6, 10, 58), "G": (6, 17, 58),
    "H<": (6, 26, 58), "H>": (6, 32, 58),
}
PATH = (
    "F<", "A>", "B", "B", "A>", "F>", "A>", "C>", "A>", "C>",
    "A>", "D",
    "H>", "H>", "H>", "H>", "H>",
    "H<", "H<", "H<", "H<", "H<",
    "C<", "C<", "C<",
    "F>", "F>", "F>", "F>",
    "D", "D",
    "C>", "C>", "C>", "C>",
    "A>", "A>", "A>", "D", "C>", "C>", "C>",
    "H>", "H>", "H>", "H>", "H>", "H>", "H>", "H>",
)
RINGS = {
    (6, 22), (7, 21), (7, 23), (8, 22),
    (15, 25), (16, 24), (16, 26), (17, 25),
}


def marker(env):
    grid = arr(env.frame())
    return tuple(
        (int(r), int(c)) for r, c in zip(*((grid == 13).nonzero()))
        if (int(r), int(c)) not in RINGS
    )


def run(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    for name in PATH:
        env.step(*BUTTONS[name])
    print("root", env.levels_completed, marker(env))
    for count in range(1, 9):
        before = env.frame()
        env.step(*BUTTONS["C<"])
        print(
            count, frame_delta(before, env.frame())["count"],
            env.levels_completed, marker(env),
        )
        if env.levels_completed > 6:
            break


arena.run_program("s5i5", run)
