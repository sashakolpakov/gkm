"""Trace the minimal level-7 gravity crossing from a pristine entry clone."""

import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components, frame_delta


ROWS = [3 + 6 * i for i in range(10)]
COLS = [15 + 6 * j for j in range(8)]
PALETTE = {0: "v", 3: "#", 5: "#", 7: "T", 8: "g", 9: "A",
           10: ".", 11: "a", 12: "s", 14: "Y", 15: "h"}


def center(frame, color):
    blobs = [
        blob for blob in connected_components(frame, colors=(color,), min_area=2)
        if blob.bbox[0] < 63
    ]
    if not blobs:
        return None
    blob = blobs[0]
    return tuple(round(value, 1) for value in blob.centroid[::-1])


def controls(frame):
    return tuple(
        round(blob.centroid[0])
        for blob in connected_components(frame, colors=(8,), min_area=3)
        if blob.bbox[0] < 63 and blob.bbox[1] <= 5
    )


def lattice(frame):
    return "/".join(
        "".join(PALETTE.get(int(frame[y][x]), "?") for x in COLS)
        for y in ROWS
    )


def state(node):
    if node.terminal():
        return node.levels_completed, True, None, None, (), ""
    frame = node.frame()
    return (
        node.levels_completed,
        False,
        center(frame, 9),
        center(frame, 7),
        controls(frame),
        lattice(frame),
    )


def signed_shift(before, after):
    a, b = np.asarray(before)[:63], np.asarray(after)[:63]
    scored = []
    for bands in range(-10, 11):
        offset = 6 * bands
        if offset >= 0:
            left, right = a[:63 - offset], b[offset:]
        else:
            left, right = a[-offset:], b[:63 + offset]
        terrain = {3: 1, 5: 1, 10: 2, 0: 3}
        matches = mismatches = 0
        for i, y in enumerate(ROWS):
            other = i + bands
            if not 0 <= other < len(ROWS):
                continue
            for x in COLS:
                ca = terrain.get(int(before[y][x]))
                cb = terrain.get(int(after[ROWS[other]][x]))
                if ca is None or cb is None:
                    continue
                matches += ca == cb
                mismatches += ca != cb
        pixels = int(np.count_nonzero(left[:, 6:] == right[:, 6:]))
        scored.append((matches - 2 * mismatches, matches, pixels,
                       -abs(bands), bands))
    score, matches, pixels, _, bands = max(scored)
    return bands, score, matches, pixels


def trace(root, label, actions):
    node = root.clone()
    print("TRACE", label, 0, None, state(node))
    net = 0
    for index, action in enumerate(actions, 1):
        if node.terminal():
            break
        before = np.asarray(node.frame()).copy()
        node.step(*action)
        shift = signed_shift(before, node.frame())
        net += shift[0]
        print("TRACE", label, index, action, "shift", shift, "net", net, state(node))


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    gravity = (6, 3, 3)
    right = (4,)
    left = (3,)
    support = (6, 39, 51)
    trace(env, "plain", [gravity, *([right] * 7)])
    trace(env, "support", [support, gravity, *([right] * 7)])
    trace(
        env,
        "seed_open",
        [right, right, right, support, gravity, right, gravity, right,
         left, (7,), (7,), left],
    )
    crossed = [
        right, right, right, support, gravity, right, gravity,
    ]
    trace(
        env,
        "direct_target",
        [*crossed, (7,), right, right, *([left] * 7)],
    )
    trace(
        env,
        "void_support",
        [*crossed, (7,), (6, 51, 51), right, right, *([left] * 7)],
    )
    trace(
        env,
        "void_remote",
        [*crossed, (7,), (6, 51, 51), (7,), right, right],
    )
    trace(
        env,
        "solve_attempt",
        [
            *crossed,
            (7,),
            (6, 51, 51),
            (7,),
            right,
            right,
            left,
            left,
            (6, 3, 5),
            left,
            left,
            left,
            left,
            (7,),
        ],
    )

    bottom = env.clone()
    bottom_path = [
        *crossed, (7,), (6, 51, 51), (7,), right, right, left, left,
    ]
    for action in bottom_path:
        bottom.step(*action)
    frame = bottom.frame()
    candidates = [
        blob for blob in connected_components(frame, colors=(8,), min_area=2)
        if blob.bbox[0] < 63
    ]
    print(
        "BOTTOM_CONTROLS",
        [(blob.bbox, blob.area, blob.centroid) for blob in candidates],
    )
    for blob in candidates:
        y, x = blob.centroid
        node = bottom.clone()
        before = np.asarray(node.frame()).copy()
        action = (6, int(round(x)), int(round(y)))
        node.step(*action)
        first = frame_delta(before[:63], node.frame()[:63])
        node.step(7)
        second = frame_delta(before[:63], node.frame()[:63])
        print(
            "BOTTOM_TRY", action,
            (first["count"], first["bbox"]),
            (second["count"], second["bbox"]),
            state(node),
        )


levels, path, error = arena.run_program("bp35", probe)
print("RESULT", levels, len(path), error)
