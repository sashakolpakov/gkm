"""Bounded symbolic level-9 clone probes using only the public arena surface."""

from collections import Counter
import json
import sys

import numpy as np

sys.path.insert(0, "/private/tmp/gkm-submitted-protocol-lf52-20260805/arc/crack_lab")

import gkm_arena as arena

from perception import arr, connected_components, frame_delta, safe_step


def world_delta(before, after):
    a, b = arr(before), arr(after)
    changed = np.argwhere(a[1:, :] != b[1:, :])
    transitions = Counter(
        (int(a[row + 1, col]), int(b[row + 1, col]))
        for row, col in changed
    )
    if not len(changed):
        return 0, None, ()
    rows = changed[:, 0] + 1
    cols = changed[:, 1]
    bbox = (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))
    return len(changed), bbox, tuple(sorted(transitions.items()))


def pieces(frame):
    blobs = connected_components(frame, colors=(1, 11, 12, 14), min_area=2)
    holes = sum(
        blob.color == 1 and blob.size == (4, 4) and blob.area == 16
        for blob in blobs
    )
    mobile = tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in blobs
        if blob.color != 1
    )
    return holes, mobile


def lattice(frame):
    f = arr(frame)
    symbols = []
    for row in range(18, 55, 6):
        line = []
        for col in range(18, 61, 6):
            values = tuple(sorted(int(v) for v in np.unique(f[row:row + 4, col:col + 4])))
            line.append({(1,): "o", (14,): "P", (12,): "C", (10,): "."}.get(values, str(values)))
        symbols.append((row, tuple(line)))
    return tuple(symbols)


def peg_positions(frame):
    return tuple(
        blob.top_left
        for blob in connected_components(frame, colors=(14,))
        if blob.size == (4, 4)
    )


def world_key(frame):
    return arr(frame)[1:, :].tobytes()


def legal_peg_moves(root):
    moves = []
    root_key = world_key(root.frame())
    for row, col in peg_positions(root.frame()):
        source = (6, col + 1, row + 1)
        selected = advance(root, (source,))
        if world_key(selected.frame()) == root_key:
            continue
        for dr, dc in ((-12, 0), (12, 0), (0, -12), (0, 12)):
            destination_row, destination_col = row + dr, col + dc
            if not (0 <= destination_row < 64 and 0 <= destination_col < 64):
                continue
            destination = (6, destination_col + 1, destination_row + 1)
            if destination[1] >= 64 or destination[2] >= 64:
                continue
            child = advance(root, (source, destination))
            if world_key(child.frame()) not in (root_key, world_key(selected.frame())):
                moves.append(((row, col), (destination_row, destination_col), pieces(child.frame())))
    return tuple(moves)


def advance(root, path):
    node = root.clone()
    for action in path:
        safe_step(node, action)
    return node


def probe(env):
    with open("checkpoint.json") as handle:
        checkpoint = json.load(handle)
    for action in checkpoint["final_path"]:
        env.step(action)

    root = env.clone()
    base = arr(root.frame()).copy()
    print("BASE", env.levels_completed, tuple(env.actions), "pieces", pieces(base))
    print("LATTICE", lattice(base))

    for action in (1, 2, 3, 4, 7):
        child = advance(root, (action,))
        print(
            "KEY", action, "level", child.levels_completed,
            "delta", world_delta(base, child.frame()),
            "pieces", pieces(child.frame()),
        )

    for action in (1, 2, 3, 4):
        node = root.clone()
        observations = []
        for count in range(1, 9):
            before = arr(node.frame()).copy()
            safe_step(node, action)
            delta = world_delta(before, node.frame())
            observations.append((count, delta[:2], pieces(node.frame())))
            if delta[0] == 0:
                break
        print("REPEAT", action, tuple(observations))

    contexts = (
        (4,), (4, 4), (4, 4, 4), (4, 1), (4, 2), (4, 3),
        (4, 7), (4, 4, 2), (4, 4, 1), (4, 4, 3),
    )
    for path in contexts:
        child = advance(root, path)
        print("PATH", path, "level", child.levels_completed, "pieces", pieces(child.frame()))

    click_points = (
        ("peg_left", (6, 19, 43)),
        ("peg_right", (6, 43, 49)),
        ("carrier", (6, 43, 37)),
        ("hole", (6, 19, 19)),
        ("nine_square", (6, 31, 19)),
        ("nine_square_low", (6, 25, 43)),
        ("background", (6, 55, 30)),
        ("border", (6, 49, 30)),
    )
    for name, click in click_points:
        child = advance(root, (click,))
        print(
            "CLICK", name, click, "level", child.levels_completed,
            "delta", world_delta(base, child.frame()),
            "pieces", pieces(child.frame()),
        )

    for path in (
        ((6, 19, 43), 7),
        ((6, 19, 43), (6, 19, 31)),
        (4, (6, 19, 43)),
        (4, (6, 43, 37)),
        (4, (6, 43, 49)),
    ):
        child = advance(root, path)
        print("CLICK_PATH", path, "level", child.levels_completed, "pieces", pieces(child.frame()))

    bridge_move = ((6, 19, 43), (6, 31, 43))
    moved = advance(root, bridge_move)
    print(
        "BRIDGE_MOVE", bridge_move, "level", moved.levels_completed,
        "delta", world_delta(base, moved.frame()), "pieces", pieces(moved.frame()),
    )
    print("BRIDGE_LATTICE", lattice(moved.frame()))
    print("LEGAL_ROOT", legal_peg_moves(root))
    print("LEGAL_MOVED", legal_peg_moves(moved))
    for path in ((4, 4, 7), bridge_move + (7,), bridge_move + (4,), bridge_move + (4, 4, 4, 4)):
        child = advance(root, path)
        print(
            "POST_PATH", path, "level", child.levels_completed,
            "pieces", pieces(child.frame()), "legal", legal_peg_moves(child),
        )

    select_left = ((6, 19, 43),)
    selected_left = advance(root, select_left)
    for action in (1, 2, 3, 4, 7):
        child = advance(root, select_left + (action,))
        print(
            "SELECT_KEY", action,
            "delta_from_selected", world_delta(selected_left.frame(), child.frame()),
            "pieces", pieces(child.frame()), "lattice", lattice(child.frame()),
        )

    for path in (
        ((6, 25, 49),),
        ((6, 25, 49), (6, 25, 37)),
        ((6, 25, 43), (6, 25, 55)),
        ((6, 31, 19),),
        ((6, 37, 25),),
    ):
        child = advance(root, path)
        print(
            "BRIDGE_OVER_BRIDGE", path, "level", child.levels_completed,
            "delta", world_delta(base, child.frame()), "lattice", lattice(child.frame()),
        )
    for path in (
        select_left + (4, (6, 31, 43)),
        select_left + (1, (6, 31, 43)),
        select_left + (2, (6, 31, 43)),
        select_left + (3, (6, 31, 43)),
        select_left + ((6, 43, 49),),
        bridge_move + ((6, 31, 43), 1),
        bridge_move + ((6, 31, 43), 2),
        bridge_move + ((6, 31, 43), 3),
        bridge_move + ((6, 31, 43), 4),
    ):
        child = advance(root, path)
        print("SELECT_CONTEXT", path, "level", child.levels_completed, "pieces", pieces(child.frame()))

    bridge_piece_move = bridge_move + ((6, 25, 43), (6, 37, 43))
    for path in (
        bridge_move + ((6, 25, 43),),
        bridge_piece_move,
        bridge_piece_move + ((6, 31, 43),),
        bridge_piece_move + ((6, 31, 43), (6, 43, 43)),
    ):
        child = advance(root, path)
        print(
            "MOVABLE_BRIDGE_CONTEXT", path, "level", child.levels_completed,
            "delta", world_delta(base, child.frame()),
            "pieces", pieces(child.frame()), "lattice", lattice(child.frame()),
        )


levels, path, error = arena.run_program("lf52", probe)
print("PROBE_RESULT", levels, len(path), error)
