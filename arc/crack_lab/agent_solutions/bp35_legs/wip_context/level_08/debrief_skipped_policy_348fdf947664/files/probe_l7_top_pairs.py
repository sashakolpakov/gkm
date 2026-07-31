"""Test ordered persistent-support pairs before the upper gravity transition."""

from itertools import permutations
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import COL_ANCHORS, ROW_ANCHORS, _cell_shape, click_action
from perception import connected_components
from probe_level7_reward_recovery import PREFIX, SUFFIX


LEFT, RIGHT, UNDO = (3,), (4,), (7,)
TOP_ROUTE = [
    *PREFIX,
    click_action(5, 2),
    *SUFFIX,
    LEFT, (6, 3, 9), RIGHT, (6, 3, 39),
    LEFT, LEFT, LEFT,
]
DROP = (6, 3, 27)


def avatar_cell(frame):
    ys, xs = np.where(np.asarray(frame) == 9)
    if not len(xs):
        return None
    y, x = float(ys.mean()), float(xs.mean())
    return (
        min(range(10), key=lambda i: abs(ROW_ANCHORS[i] - y)),
        min(range(8), key=lambda j: abs(COL_ANCHORS[j] - x)),
    )


def target_cell(frame):
    ys, xs = np.where(np.asarray(frame)[:63] == 7)
    if not len(xs):
        return None
    y, x = float(ys.mean()), float(xs.mean())
    return (
        min(range(10), key=lambda i: abs(ROW_ANCHORS[i] - y)),
        min(range(8), key=lambda j: abs(COL_ANCHORS[j] - x)),
    )


def controls(frame):
    return tuple(
        (6, round(blob.centroid[1]), round(blob.centroid[0]))
        for blob in connected_components(frame, colors=(8,), min_area=2)
        if blob.bbox[0] < 63 and blob.bbox[1] <= 5
    )


def lattice(frame):
    palette = {
        0: "v", 3: "#", 5: "#", 7: "T", 8: "g", 9: "A",
        10: ".", 11: "a", 12: "s", 14: "Y", 15: "h",
    }
    return "/".join(
        "".join(
            palette.get(int(frame[y][x]), "?") for x in COL_ANCHORS
        )
        for y in ROW_ANCHORS
    )


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)
    for action in TOP_ROUTE:
        env.step(*action)
        if env.terminal():
            print("TOP_PAIR_ROOT_DEAD", action)
            return

    base_level = int(env.levels_completed)
    root = env.clone()
    support_cells = [
        (i, j)
        for i in range(6, 10)
        for j in range(8)
        if _cell_shape(root.frame(), i, j)[0] in (12, 14)
    ]
    ordered = [(cell,) for cell in support_cells]
    ordered += list(permutations(support_cells, 2))
    outcomes = {}
    tested = 0
    for cells in ordered:
        node = root.clone()
        route = [DROP, UNDO, *[click_action(*cell) for cell in cells], RIGHT]
        for action in route:
            node.step(*action)
            if node.terminal() or node.levels_completed > base_level:
                break
        if node.levels_completed > base_level:
            print("TOP_PAIR_WIN", cells, route, flush=True)
            return
        if node.terminal():
            continue
        visible = controls(node.frame())
        if visible:
            node.step(*min(
                visible,
                key=lambda action: abs(action[2] - ROW_ANCHORS[6]),
            ))
        if node.levels_completed > base_level:
            print("TOP_PAIR_WIN", cells, route, "gravity", flush=True)
            return
        if node.terminal():
            continue
        positions = [avatar_cell(node.frame())]
        for _ in range(5):
            node.step(*RIGHT)
            positions.append(avatar_cell(node.frame()))
            if node.levels_completed > base_level:
                print("TOP_PAIR_WIN", cells, route, "rights", flush=True)
                return
            if node.terminal():
                break
        tested += 1
        if node.terminal():
            continue
        state = (
            avatar_cell(node.frame()),
            target_cell(node.frame()),
            controls(node.frame()),
            lattice(node.frame()),
        )
        prior = outcomes.get(state)
        if prior is None:
            outcomes[state] = (cells, tuple(positions), 1)
        else:
            outcomes[state] = (prior[0], prior[1], prior[2] + 1)

    ranked = sorted(
        outcomes.items(),
        key=lambda item: (
            item[0][1] is None,
            -(item[0][0] or (0, 0))[1],
            item[0][0] or (99, 99),
        ),
    )
    print(
        "TOP_PAIR_DONE", tested, len(ordered), len(outcomes),
        support_cells, flush=True,
    )
    for state, witness in ranked[:30]:
        print("TOP_PAIR_STATE", witness, state, flush=True)


arena.run_program("bp35", probe)
