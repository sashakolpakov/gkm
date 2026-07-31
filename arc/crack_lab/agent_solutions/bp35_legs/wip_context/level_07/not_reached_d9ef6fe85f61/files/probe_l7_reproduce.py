"""Compact clean-room probes for the pristine bp35 level-7 entry."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import color_counts, connected_components, frame_delta


ROWS = [3 + 6 * i for i in range(10)]
COLS = [15 + 6 * j for j in range(8)]


def compact_components(frame):
    return [
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, min_area=3)
        if blob.area < 3000
    ]


def lattice(frame):
    return tuple(tuple(int(frame[y][x]) for x in COLS) for y in ROWS)


def lattice_delta(before, after):
    a, b = lattice(before), lattice(after)
    return [
        (i, j, a[i][j], b[i][j])
        for i in range(10)
        for j in range(8)
        if a[i][j] != b[i][j]
    ]


def avatar_cells(frame):
    return [
        (i, j)
        for i, y in enumerate(ROWS)
        for j, x in enumerate(COLS)
        if int(frame[y][x]) in (9, 11)
    ]


def avatar_position(frame):
    blobs = connected_components(frame, colors=(9,), min_area=2)
    if not blobs:
        return None
    blob = blobs[0]
    return tuple(round(value, 1) for value in blob.centroid[::-1])


def object_centers(frame, color):
    return [
        (tuple(round(value, 1) for value in blob.centroid[::-1]), blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(color,), min_area=2)
        if blob.bbox[0] < 63
    ]


def encoded_lattice(frame):
    alphabet = {0: "v", 3: "#", 5: "#", 7: "T", 8: "g", 9: "A",
                10: ".", 11: "a", 12: "s", 14: "Y", 15: "h"}
    return "/".join(
        "".join(alphabet.get(value, "?") for value in row)
        for row in lattice(frame)
    )


def support_shapes(frame):
    out = []
    for i, y in enumerate(ROWS):
        for j, x in enumerate(COLS):
            if int(frame[y][x]) != 12:
                continue
            r0, c0 = 6 * i, 13 + 6 * j
            area = sum(
                int(frame[r][c]) == 12
                for r in range(r0, min(63, r0 + 6))
                for c in range(c0, c0 + 6)
            )
            out.append((i, j, area))
    return out


def state_summary(label, node):
    frame = node.frame()
    print(
        "STATE", label, node.levels_completed, node.terminal(),
        avatar_position(frame), object_centers(frame, 7),
        encoded_lattice(frame), support_shapes(frame),
    )


def count_delta(before, after):
    a, b = color_counts(before), color_counts(after)
    return {color: b.get(color, 0) - a.get(color, 0) for color in set(a) | set(b)
            if a.get(color, 0) != b.get(color, 0)}


def sequence_summary(env, label, sequence):
    node = env.clone()
    snapshots = [node.frame()]
    for action in sequence:
        node.step(*action)
        snapshots.append(node.frame())
    transitions = []
    for before, after in zip(snapshots, snapshots[1:]):
        delta = frame_delta(before, after)
        transitions.append((
            {"count": delta["count"], "bbox": delta["bbox"]},
            count_delta(before, after),
            avatar_cells(after),
            lattice_delta(before, after),
        ))
    print("SEQ", label, node.levels_completed, node.terminal(), transitions)


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    before = env.frame()
    print("ENTRY", env.levels_completed, env.terminal(), list(env.actions))
    print("COLORS", color_counts(before))
    print("OBJECTS", compact_components(before))
    print("LATTICE", lattice(before))
    for action in env.actions:
        clone = env.clone()
        clone.step(action)
        print(
            "SCALAR",
            action,
            clone.levels_completed,
            clone.terminal(),
            frame_delta(before, clone.frame()),
        )

    targets = [
        ("control", 3, 3),
        ("open", 15, 3),
        ("support", 27, 3),
        ("hazard", 21, 27),
        ("avatar", 21, 39),
        ("wall", 15, 57),
    ]
    for action in (6, 7):
        for name, x, y in targets:
            clone = env.clone()
            clone.step(action, x, y)
            after = clone.frame()
            delta = frame_delta(before, after)
            print(
                "COORD",
                action,
                name,
                (x, y),
                clone.levels_completed,
                clone.terminal(),
                avatar_cells(after),
                {"count": delta["count"], "bbox": delta["bbox"]},
                count_delta(before, after),
                lattice_delta(before, after),
            )

    sequences = {
        "support_then_7": [(6, 27, 3), (7,)],
        "support_then_7_coord": [(6, 27, 3), (7, 15, 3)],
        "support_twice": [(6, 27, 3), (6, 27, 3)],
        "support_then_other": [(6, 27, 3), (6, 33, 3), (7,)],
        "control_then_7": [(6, 3, 3), (7,)],
        "control_then_7_coord": [(6, 3, 3), (7, 55, 55)],
        "inert_then_7": [(6, 15, 3), (7,)],
    }
    for label, sequence in sequences.items():
        sequence_summary(env, label, sequence)

    state_summary("base", env)
    gravity = env.clone()
    gravity.step(6, 3, 3)
    state_summary("gravity", gravity)
    for action in ((3,), (4,)):
        node = gravity.clone()
        node.step(*action)
        state_summary(f"gravity_{action[0]}", node)
    targets = object_centers(gravity.frame(), 7)
    if targets:
        (x, y), _, _ = targets[0]
        for sequence in (
            [(6, int(round(x)), int(round(y)))],
            [(7, int(round(x)), int(round(y)))],
            [(6, int(round(x)), int(round(y))), (7,)],
        ):
            node = gravity.clone()
            for action in sequence:
                node.step(*action)
            state_summary(f"target_{sequence}", node)


levels, path, error = arena.run_program("bp35", probe)
print("RESULT", levels, len(path), error)
