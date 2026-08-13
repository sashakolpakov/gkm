"""Reveal level 9 after bounded visually-idle 1/2 control prefixes."""

import itertools
import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, safe_step


OPENING = (
    ((42, 18), (42, 30)), ((48, 24), (36, 24)),
    ((42, 24), (30, 24)), ((36, 24), (24, 24)),
    ((30, 24), (18, 24)), ((18, 24), (18, 36)),
    ((18, 36), (30, 36)), ((24, 36), (36, 36)),
    ((30, 36), (42, 36)), ((36, 36), (48, 36)),
    ((48, 42), (48, 30)), ((48, 30), (36, 30)),
    ((48, 36), (36, 36)), ((36, 30), (36, 42)),
)


def move(node, source, destination):
    safe_step(node, (6, source[1] + 1, source[0] + 1))
    safe_step(node, (6, destination[1] + 1, destination[0] + 1))


def pieces(frame):
    blobs = connected_components(frame, colors=(9, 11, 12, 14, 15))
    return tuple(sorted(
        (blob.color, blob.bbox, blob.area)
        for blob in blobs
        if blob.color != 9 or blob.area >= 12
    ))


def prefixes():
    out = {()}
    for action in (1, 2, 3):
        for count in range(1, 13):
            out.add((action,) * count)
    for length in range(2, 9):
        out.add(tuple(1 if index % 2 == 0 else 2
                      for index in range(length)))
        out.add(tuple(2 if index % 2 == 0 else 1
                      for index in range(length)))
    for first, second in itertools.product(range(1, 5), repeat=2):
        out.add((1,) * first + (2,) * second)
        out.add((2,) * first + (1,) * second)
    return tuple(sorted(out, key=lambda path: (len(path), path)))


def reveal_paths():
    out = {("lead", prefix): prefix + (4,) * 9 for prefix in prefixes()}
    for offset in range(10):
        for action in (1, 2):
            for count in range(1, 5):
                label = ("insert", offset, action, count)
                out[label] = ((4,) * offset + (action,) * count
                              + (4,) * (9 - offset))
    out[("between", 1)] = tuple(
        action for _ in range(9) for action in (4, 1)
    )
    out[("between", 2)] = tuple(
        action for _ in range(9) for action in (4, 2)
    )
    out[("between", 12)] = tuple(
        action for index in range(9)
        for action in (4, 1 if index % 2 == 0 else 2)
    )
    return tuple(out.items())


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    for action in campaign:
        safe_step(env, tuple(action) if isinstance(action, list) else action)
    before_opening = os.environ.get("L9_BEFORE_OPENING") == "1"
    if not before_opening:
        for source, destination in OPENING:
            move(env, source, destination)

    layouts = {}
    paths = ((prefix, prefix) for prefix in prefixes()) if before_opening else reveal_paths()
    for label, path in paths:
        child = env.clone()
        for action in path:
            safe_step(child, action)
        if before_opening:
            for source, destination in OPENING:
                move(child, source, destination)
            for _ in range(9):
                safe_step(child, 4)
        layout = (arr(child.frame())[1:, :].tobytes(), pieces(child.frame()))
        layouts.setdefault(layout, (label, path))
    print("hidden_layouts", len(prefixes()) if before_opening else len(paths),
          len(layouts), flush=True)
    for layout, details in sorted(
            layouts.items(), key=lambda item: (len(item[1][1]), item[1][1])):
        print("hidden_layout", details, layout[1], flush=True)


arena.run_program("lf52", probe)
