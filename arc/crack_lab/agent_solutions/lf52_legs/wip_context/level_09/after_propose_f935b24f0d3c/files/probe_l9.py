"""Compact, bounded observations from the verified level-9 entry state."""

import json
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import (
    action_deltas, arr, color_counts, connected_components, replay, safe_step,
)


def component_summary(frame):
    grouped = defaultdict(list)
    for blob in connected_components(frame):
        grouped[(blob.color, blob.size, blob.area)].append(blob.top_left)
    for signature, positions in sorted(grouped.items()):
        print("COMP", signature, "n=", len(positions), "at", positions)


def delta_summary(before, after):
    a, b = arr(before), arr(after)
    changed = a != b
    ys, xs = changed.nonzero()
    if not len(ys):
        return "unchanged"
    transitions = Counter((int(a[y, x]), int(b[y, x])) for y, x in zip(ys, xs))
    return {
        "n": int(len(ys)),
        "bbox": (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())),
        "transitions": sorted(transitions.items()),
    }


def ascii_crop(frame, bounds=(14, 14, 56, 58)):
    glyph = {0: " ", 1: ".", 5: "#", 9: "X", 10: " ",
             11: "B", 12: "C", 14: "O"}
    r0, c0, r1, c1 = bounds
    data = arr(frame)
    return "\n".join(
        "".join(glyph[int(value)] for value in data[row, c0:c1])
        for row in range(r0, r1)
    )


def observe(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)

    frame = env.frame()
    print("ENTRY", {"level": env.levels_completed, "actions": tuple(env.actions),
                    "shape": arr(frame).shape, "colors": color_counts(frame)})
    component_summary(frame)
    print("ENTRY_MAP\n" + ascii_crop(frame))
    print("KEY_DELTAS")
    for action, delta in action_deltas(env).items():
        clone = env.clone()
        clone.step(action)
        print(action, delta_summary(frame, clone.frame()),
              "level", clone.levels_completed)
        if action == 4:
            print("AFTER_4_MAP\n" + ascii_crop(clone.frame()))

    probes = {
        "hole_18_18": ((6, 19, 19),),
        "x_18_30": ((6, 31, 19),),
        "carrier_36_42": ((6, 43, 37),),
        "peg_42_18": ((6, 19, 43),),
        "peg_48_42": ((6, 43, 49),),
        "boundary_x_42_42": ((6, 43, 43),),
        "boundary_x_down_over_peg": ((6, 43, 43), (6, 43, 55)),
        "right_peg_up_over_boundary_x": ((6, 43, 49), (6, 43, 37)),
        "wall_16_16": ((6, 16, 16),),
        "background_10_10": ((6, 10, 10),),
        "peg1_over_x": ((6, 19, 43), (6, 31, 43)),
        "peg1_over_x_then_x_over_peg": (
            (6, 19, 43), (6, 31, 43),
            (6, 25, 43), (6, 37, 43),
        ),
    }
    print("COORDINATE_CONTEXTS")
    for name, path in probes.items():
        node = replay(env, path)
        print(name, path, delta_summary(frame, node.frame()),
              "colors", color_counts(node.frame()), "level", node.levels_completed)

    print("PEG_THEN_KEYS")
    for key_action in (1, 2, 3, 4, 7):
        path = ((6, 19, 43), key_action)
        node = replay(env, path)
        print(key_action, delta_summary(frame, node.frame()),
              "colors", color_counts(node.frame()))

    selected = replay(env, ((6, 19, 43),))
    print("SELECTION_COMPONENTS")
    for blob in connected_components(selected.frame(), colors=(2, 3)):
        print(blob)


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path), "error": str(error)})
