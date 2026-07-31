"""Bounded symbolic clone probes for level 2."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import assemble_telescoping_chain
from perception import arr, connected_components, frame_delta


COLORS = (8, 12, 9, 14)


def state(env):
    blobs = connected_components(env.frame(), min_area=1)
    top = {
        color: [(b.bbox, b.area) for b in blobs if b.color == color and b.bbox[2] < 53]
        for color in COLORS
    }
    guide = {
        color: [(b.bbox, b.area) for b in blobs if b.color == color and b.bbox[0] > 53]
        for color in COLORS
    }
    heads = [
        (b.bbox, b.area)
        for b in blobs
        if b.color == 0 and b.bbox[2] < 53
    ]
    return {
        "level": int(env.levels_completed),
        "head": heads,
        "top": top,
        "guide": guide,
    }


def apply(node, action):
    before = arr(node.frame()).copy()
    if isinstance(action, tuple):
        node.step(*action)
    else:
        node.step(action)
    delta = frame_delta(before, node.frame())
    return (delta["count"], delta["bbox"])


def probe(env):
    assemble_telescoping_chain(env)
    print("ENTRY", state(env))

    sequences = {
        "align_extend": [1, 1, 1, 4, 4, 4, 4, 4, 4, 4],
        "align_extend_retract": [1, 1, 1, 4, 4, 4, 4, 3, 3, 3, 3],
        "up_down": [1, 2],
        "extra7_entry": [7],
        "extra7_aligned": [1, 1, 1, 7],
        "extra7_contact": [1, 1, 1, 4, 4, 4, 7],
    }
    for name, actions in sequences.items():
        node = env.clone()
        print("SEQ", name)
        for index, action in enumerate(actions, 1):
            delta = apply(node, action)
            print(index, action, delta, state(node))

    clicks = [
        ("empty", (6, 20, 20)),
        ("head", (6, 8, 44)),
        ("block14", (6, 31, 27)),
        ("block8", (6, 49, 27)),
        ("guide8", (6, 25, 59)),
        ("guide14", (6, 43, 59)),
    ]
    print("CLICKS")
    for name, action in clicks:
        node = env.clone()
        print(name, action, apply(node, action), state(node))


levels, path, error = arena.run_program("sk48", probe)
print("RUN", levels, len(path), error)
