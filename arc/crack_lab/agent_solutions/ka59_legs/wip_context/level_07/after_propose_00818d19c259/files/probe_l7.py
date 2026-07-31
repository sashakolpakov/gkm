import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from perception import arr, color_counts, connected_components, frame_delta


def world(frame):
    return arr(frame)[:63]


def blobs(frame):
    f = world(frame)
    counts = color_counts(f)
    bg = max(counts, key=counts.get)
    return bg, [
        (b.color, b.bbox, b.area)
        for b in connected_components(f, min_area=2)
        if b.color != bg
    ]


def delta(a, b):
    d = frame_delta(world(a), world(b))
    transitions = {}
    x, y = world(a), world(b)
    rs, cs = (x != y).nonzero()
    for r, c in zip(rs, cs):
        pair = (int(x[r, c]), int(y[r, c]))
        transitions[pair] = transitions.get(pair, 0) + 1
    return d["count"], d["bbox"], tuple(sorted(transitions.items()))


def probe(env):
    for level in range(1, 7):
        getattr(players, f"play_level_{level}")(env)
    base = env.frame()
    bg, objects = blobs(base)
    print("level", env.levels_completed, "actions", env.actions)
    print("counts", color_counts(world(base)), "bg", bg)
    print("objects", objects)
    for action in env.actions:
        node = env.clone()
        node.step(action)
        print("action", action, delta(base, node.frame()), "objects", blobs(node.frame())[1])
    for action in (1, 2, 3, 4):
        node = env.clone()
        trace = []
        prev = node.frame()
        for i in range(1, 5):
            node.step(action)
            trace.append((i, delta(prev, node.frame()), blobs(node.frame())[1]))
            prev = node.frame()
        print("repeat", action, trace)
    # Click a confirmed pixel of every compact component, then apply each arrow.
    for color, bbox, area in objects:
        r0, c0, r1, c1 = bbox
        if area > 100 or r1 - r0 > 20 or c1 - c0 > 20:
            continue
        f = world(base)
        point = next(
            (int(c), int(r))
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if int(f[r, c]) == color
        )
        effects = {}
        for action in (1, 2, 3, 4):
            node = env.clone()
            node.step(6, *point)
            before = node.frame()
            node.step(action)
            effects[action] = delta(before, node.frame())
        print("select", (color, bbox, area, point), effects)


levels, path, err = A.run_program("ka59", probe)
print("result", levels, len(path), err)
