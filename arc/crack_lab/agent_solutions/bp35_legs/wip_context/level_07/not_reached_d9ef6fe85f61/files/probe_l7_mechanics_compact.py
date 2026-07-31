"""Compact, fresh tests of level-7 support selection and moving objects."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena
import gkm_legs as campaign

from legs import COL_ANCHORS, ROW_ANCHORS, _cell_shape, click_action
from probe_l7_raw_search import SEED


if campaign._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


PALETTE = {
    0: "v", 3: "#", 5: "D", 7: "T", 8: "g", 9: "A",
    10: ".", 11: "a", 12: "s", 14: "Y", 15: "h",
}


def cells(frame, color):
    return tuple(
        (i, j, _cell_shape(frame, i, j)[1])
        for i, y in enumerate(ROW_ANCHORS)
        for j, x in enumerate(COL_ANCHORS)
        if int(frame[y][x]) == color
    )


def lattice(frame):
    return "/".join(
        "".join(PALETTE.get(int(frame[y][x]), "?") for x in COL_ANCHORS)
        for y in ROW_ANCHORS
    )


def crop(frame, i, j, color):
    r0, c0 = 6 * i, 13 + 6 * j
    return "/".join(
        "".join("x" if int(frame[r][c]) == color else "." for c in range(c0, c0 + 6))
        for r in range(r0, r0 + 6)
    )


def state(node):
    if node.terminal():
        return ("dead",)
    frame = node.frame()
    return (
        "alive",
        cells(frame, 9),
        cells(frame, 12),
        cells(frame, 0),
        cells(frame, 7),
        lattice(frame),
    )


def replay(root, actions):
    node = root.clone()
    for action in actions:
        node.step(*action)
        if node.terminal():
            break
    return node


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)
    root = env.clone()
    frame = root.frame()
    print("ENTRY", state(root))
    for color in (9, 12, 15):
        for i, j, area in cells(frame, color):
            print("SHAPE", color, (i, j, area), crop(frame, i, j, color))

    supports = [(i, j) for i, j, _ in cells(frame, 12)]
    tests = [
        ("wait", [(7,)] * int(os.environ.get("WAIT_COUNT", "4"))),
        ("move_undo", [(4,), (7,), (7,)]),
        ("move_undo_branch", [(4,), (7,), (4,), (6, 9, 3), (4,)]),
        ("move2_undo", [(4,), (4,), (7,), (7,), (7,)]),
        ("one", [click_action(*supports[0]), (7,), (7,), (7,)]),
        (
            "support_undo_branch",
            [click_action(*supports[-1]), (7,), (4,), (6, 9, 3), (4,)],
        ),
        (
            "support_move_undo",
            [click_action(*supports[-1]), (4,), (7,), (7,), (7,)],
        ),
        (
            "two",
            [
                click_action(*supports[0]),
                click_action(*supports[-1]),
                (7,), (7,), (7,),
            ],
        ),
        ("flip", [(6, 3, 3)]),
        ("flip_move_undo", [(6, 3, 3), (4,), (7,), (7,), (7,)]),
        ("flip_wait", [(6, 3, 3), (7,), (7,), (7,)]),
    ]
    for name, actions in tests:
        node = root.clone()
        print("TEST", name, 0, state(node))
        for index, action in enumerate(actions, 1):
            node.step(*action)
            print("TEST", name, index, action, state(node))
            if node.terminal():
                break
        if not node.terminal():
            moved = cells(node.frame(), 0)
            for i, j, area in moved:
                print("DARK_SHAPE", name, (i, j, area), crop(node.frame(), i, j, 0))

    node = root.clone()
    held = []
    for index, action in enumerate(SEED, 1):
        before = node.frame()
        event = None
        if action[0] == 6:
            _, x, y = action
            if x <= 5:
                event = ("left", int(before[y][x]) if y < 63 else -1, y)
            else:
                i = min(range(10), key=lambda k: abs(ROW_ANCHORS[k] - y))
                j = min(range(8), key=lambda k: abs(COL_ANCHORS[k] - x))
                event = ("cell", i, j, *_cell_shape(before, i, j))
            held.append(event)
        elif action[0] == 7:
            event = held.pop() if held else None
        node.step(*action)
        if action[0] in (6, 7):
            print(
                "SEED_EVENT", index, action,
                "push" if action[0] == 6 else "pop", event,
                "held", len(held), "state", state(node)[:5],
            )
        if node.terminal():
            break


arena.run_program("bp35", probe)
