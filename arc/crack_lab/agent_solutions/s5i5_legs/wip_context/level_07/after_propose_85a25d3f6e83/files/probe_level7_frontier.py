import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, frame_delta


CONTROLS = (
    ("A<", (6, 4, 51)), ("A>", (6, 10, 51)), ("B", (6, 17, 51)),
    ("C<", (6, 26, 51)), ("C>", (6, 32, 51)), ("D", (6, 39, 51)),
    ("E<", (6, 54, 51)), ("E>", (6, 60, 51)),
    ("F<", (6, 4, 58)), ("F>", (6, 10, 58)), ("G", (6, 17, 58)),
    ("H<", (6, 26, 58)), ("H>", (6, 32, 58)), ("I", (6, 60, 58)),
)
LOOKUP = dict(CONTROLS)
RINGS = {
    (6, 22), (7, 21), (7, 23), (8, 22),
    (15, 25), (16, 24), (16, 26), (17, 25),
}
FRONTIER = (
    "F<", "A>", "B", "B", "A>", "F>", "A>", "C>", "A>", "C>",
    "A>", "G", "C>", "C>",
    "H>", "H>", "H>", "H>", "H>", "H>", "H>",
)


def markers(frame):
    grid = arr(frame)
    return tuple(
        (int(r), int(c)) for r, c in zip(*((grid[:48] == 13).nonzero()))
        if (int(r), int(c)) not in RINGS
    )


def summary(env):
    bodies = tuple(
        (b.color, b.bbox) for b in connected_components(
            env.frame(), colors={8, 9, 10, 11, 12, 14}, min_area=4
        ) if b.bbox[0] < 42
    )
    return markers(env.frame()), bodies


def run(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    for name in FRONTIER:
        env.step(*LOOKUP[name])
    print("frontier", summary(env))
    for name, action in CONTROLS:
        node = env.clone()
        states = []
        for _ in range(6):
            before = node.frame()
            node.step(*action)
            delta = frame_delta(before, node.frame())["count"]
            states.append((delta, markers(node.frame())))
            if node.levels_completed > 6 or delta == 0:
                break
        print(name, states, "level", node.levels_completed)


arena.run_program("s5i5", run)
