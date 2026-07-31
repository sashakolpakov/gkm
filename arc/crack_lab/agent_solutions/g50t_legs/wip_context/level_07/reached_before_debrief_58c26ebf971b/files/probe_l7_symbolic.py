"""Compact observational traces for g50t level 7."""
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _avatar_pos
from perception import connected_components


FIRST = [2, 1, 2, 1, 2, 2, 3, 5, 2, 1, 2, 1, 2]
KNOWN = (
    FIRST
    + [1, 3, 3, 1, 1, 5]
    + [2, 1, 2, 1, 2, 2, 3, 5]
    + [2, 1, 2, 2, 3, 5]
    + [2, 1] * 5
    + [2, 1] * 5
    + [3, 3, 1, 1, 5]
    + [2]
)


def compact(env):
    try:
        frame = np.asarray(env.frame())
        helper = next(
            (b.bbox[:2] for b in connected_components(
                frame, colors=(14,), min_area=4)),
            None,
        )
        marker = next(
            (b.bbox[:2] for b in connected_components(
                frame, colors=(9,), min_area=4)
             if b.bbox[0] == 1),
            None,
        )
        areas = tuple(
            (
                color,
                sum(b.area for b in connected_components(
                    frame, colors=(color,), min_area=1)),
            )
            for color in (11, 15)
        )
        return (
            int(env.levels_completed),
            bool(env.terminal()),
            _avatar_pos(frame),
            marker,
            helper,
            areas,
        )
    except (IndexError, ValueError):
        return ("bad",)


def trace(root, label, actions):
    node = root.clone()
    print(label, 0, compact(node), flush=True)
    for tick, action in enumerate(actions, 1):
        if node.terminal():
            break
        try:
            node.step(action)
        except (IndexError, ValueError):
            print(label, tick, action, ("bad-step",), flush=True)
            break
        print(label, tick, action, compact(node), flush=True)


def macro_trace(root, label, macro, count):
    node = root.clone()
    print(label, 0, compact(node), flush=True)
    for cycle in range(1, count + 1):
        for action in macro:
            if node.terminal():
                break
            try:
                node.step(action)
            except (IndexError, ValueError):
                print(label, cycle, ("bad-step",), flush=True)
                return
        print(label, cycle, compact(node), flush=True)
        if node.terminal():
            break


def probe(env):
    with open("checkpoint.json") as fh:
        checkpoint = json.load(fh)["final_path"]
    for action in checkpoint:
        env.step(action)
    trace(env, "pump", [2, 1] * 8)
    trace(env, "first", FIRST)
    trace(env, "known", KNOWN)
    macro_trace(env, "repeat-bottom", [2, 2, 3, 5], 20)
    trace(env, "latch32", FIRST + [5] + [2, 1] * 10)
    trace(env, "latch26", FIRST + [1, 5] + [2, 1] * 10)


levels, path, err = arena.run_program("g50t", probe)
print("result", levels, len(path), err)
