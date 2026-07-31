import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, frame_delta


CONTROLS = {
    "A<": (6, 4, 51), "A>": (6, 10, 51), "B": (6, 17, 51),
    "C<": (6, 26, 51), "C>": (6, 32, 51), "D": (6, 39, 51),
    "E<": (6, 54, 51), "E>": (6, 60, 51),
    "F<": (6, 4, 58), "F>": (6, 10, 58), "G": (6, 17, 58),
    "H<": (6, 26, 58), "H>": (6, 32, 58), "I": (6, 60, 58),
}
PREFIX = (
    "F<", "A>", "B", "B", "A>", "F>", "A>", "C>", "A>", "C>",
    "A>", "D", "H>", "H>", "H>", "H>", "H>",
)
RINGS = {
    (6, 22), (7, 21), (7, 23), (8, 22),
    (15, 25), (16, 24), (16, 26), (17, 25),
}


def summary(env):
    grid = arr(env.frame())
    markers = tuple(
        (int(r), int(c)) for r, c in zip(*((grid[:42] == 13).nonzero()))
        if (int(r), int(c)) not in RINGS
    )
    bodies = tuple(
        (b.color, b.bbox) for b in connected_components(
            env.frame(), colors={9, 11, 12, 14}, min_area=4
        ) if b.bbox[0] < 42
    )
    return markers, bodies


def run(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    for name in PREFIX:
        env.step(*CONTROLS[name])
    print("root", summary(env))
    stages = (
        ("H<", 6), ("C<", 6), ("F>", 4), ("D", 2),
        ("C>", 4), ("I", 4), ("G", 4), ("H>", 4), ("H<", 4),
    )
    for name, times in stages:
        for index in range(times):
            before = env.frame()
            env.step(*CONTROLS[name])
            changed = frame_delta(before, env.frame())["count"]
            print(name, index + 1, changed, summary(env),
                  "level", env.levels_completed)
            if env.levels_completed > 6 or changed == 0:
                break
        if env.levels_completed > 6:
            break


arena.run_program("s5i5", run)
