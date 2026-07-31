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
    for retreat in range(7):
        root = env.clone()
        for _ in range(retreat):
            root.step(*CONTROLS["H<"])
        print("retreat", retreat, summary(root))
        for name in ("B", "D", "G"):
            node = root.clone()
            before = node.frame()
            node.step(*CONTROLS[name])
            changed = frame_delta(before, node.frame())["count"]
            print(" ", name, changed, summary(node))

    folded = env.clone()
    for _ in range(5):
        folded.step(*CONTROLS["H<"])
    for _ in range(3):
        folded.step(*CONTROLS["C<"])
    print("folded", summary(folded))
    for shift in range(8):
        root = folded.clone()
        for _ in range(shift):
            root.step(*CONTROLS["F>"])
        states = []
        for _ in range(4):
            before = root.frame()
            root.step(*CONTROLS["D"])
            states.append((frame_delta(before, root.frame())["count"],
                           summary(root)))
        print("shift", shift, states)


arena.run_program("s5i5", run)
