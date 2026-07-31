import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, frame_delta


CONTROLS = {
    "E<": (6, 54, 51), "E>": (6, 60, 51), "I": (6, 60, 58),
}
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
            env.frame(), colors={8, 10}, min_area=4
        ) if b.bbox[0] < 42
    )
    return markers, bodies


def run(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    for direction in ("E<", "E>"):
        for shift in range(5):
            node = env.clone()
            for _ in range(shift):
                node.step(*CONTROLS[direction])
            states = []
            for _ in range(5):
                before = node.frame()
                node.step(*CONTROLS["I"])
                states.append((
                    frame_delta(before, node.frame())["count"],
                    summary(node),
                ))
            print(direction, shift, summary(node), states)


arena.run_program("s5i5", run)
