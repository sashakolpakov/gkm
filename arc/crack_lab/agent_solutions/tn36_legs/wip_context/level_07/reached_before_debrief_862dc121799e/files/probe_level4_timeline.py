"""Observe selectable protocol examples while only advancing their timer."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr, connected_components


def demo_signature(node):
    frame = arr(node.frame())
    objects = tuple(
        (blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(4,))
        if blob.bbox[1] < 31
    )
    colors = tuple(
        (color, int((frame[3:32, 1:31] == color).sum()))
        for color in sorted(set(int(value) for value in frame[3:32, 1:31].flat))
    )
    return objects, colors


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    for selector in (5, 15, 25, 35):
        clone = env.clone()
        clone.step(6, selector, 58)
        previous = demo_signature(clone)
        print("selector", selector, "tick", 0, previous)
        for tick in range(1, 71):
            clone.step(6, 30, 30)
            current = demo_signature(clone)
            timer = int((arr(clone.frame())[1, :] == 9).sum())
            if current != previous or tick in (1, 15, 30, 45, 60, 61, 62, 70):
                print(
                    "selector",
                    selector,
                    "tick",
                    tick,
                    {"timer": timer, "terminal": clone.terminal(), "demo": current},
                )
            previous = current


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})
