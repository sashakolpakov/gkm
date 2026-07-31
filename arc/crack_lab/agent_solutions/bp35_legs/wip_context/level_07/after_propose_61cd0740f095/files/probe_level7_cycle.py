import json
import os
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import (
    COL_ANCHORS, ROW_ANCHORS, _cell_shape, band_shift, click_action,
    moves_used, run_actions,
)
from perception import arr


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


GRAVITY = (6, 3, 3)
PREFIX = [
    (4,), (4,), (4,), click_action(8, 4), GRAVITY, (4,), GRAVITY, (4,),
    (3,), (3,), click_action(8, 3), GRAVITY, (3,), (6, 3, 9), (3,),
    (4,), (4,), click_action(8, 2), (6, 3, 9),
    (3,), (3,), (6, 3, 15), (3,), (3,),
    (4,), click_action(7, 2), (6, 3, 21), (4,), (6, 3, 15),
]
CYCLE = [(3,), "gravity", (4,), "gravity"]


def avatar_cell(frame):
    ys, xs = (arr(frame) == 9).nonzero()
    if not len(ys):
        return None
    return int(round(ys.mean())), int(round(xs.mean()))


def edge_rows(frame):
    return [int(y) for y in range(63) if int(frame[y][3]) == 8]


def gravity_action(frame):
    return next(
        ((6, 3, y) for y in ROW_ANCHORS if int(frame[y][3]) == 8),
        None,
    )


def lattice(frame):
    palette = {
        3: "#", 5: "#", 8: "g", 9: "A", 10: ".", 11: "A",
        12: "c", 14: "Y", 15: "f",
    }
    return "/".join(
        "".join(palette.get(int(frame[y][x]), "?") for x in COL_ANCHORS)
        for y in ROW_ANCHORS
    )


def shaped_cells(frame):
    return [
        (i, j, _cell_shape(frame, i, j))
        for i in range(10) for j in range(8)
        if _cell_shape(frame, i, j)[0] not in (3, 5, 10)
    ]


def avatar_lattice(frame):
    ys, xs = (arr(frame) == 9).nonzero()
    if not len(ys):
        return None
    return (
        min(range(10), key=lambda i: abs(ROW_ANCHORS[i] - int(ys.mean()))),
        min(range(8), key=lambda j: abs(COL_ANCHORS[j] - int(xs.mean()))),
    )


def local_actions(frame):
    avatar = avatar_lattice(frame)
    actions = [(3,), (4,)]
    if avatar is None:
        return actions
    ai, aj = avatar
    for i in range(max(0, ai - 2), min(10, ai + 3)):
        for j in range(max(0, aj - 1), min(8, aj + 2)):
            color, area = _cell_shape(frame, i, j)
            if color in (12, 14) and area < 21:
                actions.append(click_action(i, j))
    gravity = gravity_action(frame)
    if gravity is not None:
        actions.append(gravity)
    return actions


def next_live_rise(root, max_states=300, max_depth=10):
    base = arr(root.frame()).copy()
    queue = deque([(root.clone(), [])])
    seen = {(base[:63].tobytes(), moves_used(base) % 2)}
    expanded = 0
    while queue and expanded < max_states:
        node, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for action in local_actions(node.frame()):
            child = node.clone()
            child.step(*action)
            expanded += 1
            child_path = path + [action]
            if child.levels_completed > 6:
                return child_path, child, expanded
            if child.terminal() or avatar_lattice(child.frame()) is None:
                continue
            if band_shift(base, child.frame()) > 0:
                return child_path, child, expanded
            frame = arr(child.frame())
            key = (frame[:63].tobytes(), moves_used(frame) % 2)
            if key not in seen:
                seen.add(key)
                queue.append((child, child_path))
            if expanded >= max_states:
                break
    return [], root.clone(), expanded


def best_live_progress(root, max_states=300, max_depth=12):
    base = arr(root.frame()).copy()
    queue = deque([(root.clone(), [])])
    seen = {(base[:63].tobytes(), moves_used(base) % 2)}
    best = (0, [], root.clone())
    expanded = 0
    while queue and expanded < max_states:
        node, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for action in local_actions(node.frame()):
            child = node.clone()
            child.step(*action)
            expanded += 1
            child_path = path + [action]
            if child.levels_completed > 6:
                return child_path, child, expanded
            if child.terminal() or avatar_lattice(child.frame()) is None:
                continue
            gain = band_shift(base, child.frame())
            if gain > best[0]:
                best = (gain, child_path, child)
            frame = arr(child.frame())
            key = (frame[:63].tobytes(), moves_used(frame) % 2)
            if key not in seen:
                seen.add(key)
                queue.append((child, child_path))
            if expanded >= max_states:
                break
    return best[1], best[2], expanded


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    print("START", env.levels_completed, env.actions, avatar_cell(env.frame()))

    node = env.clone()
    prefix_gain = 0
    for index, action in enumerate(PREFIX):
        before = arr(node.frame()).copy()
        node.step(*action)
        gain = band_shift(before, node.frame())
        prefix_gain += gain
        if gain:
            print(
                "RISE", index + 1, action, gain, prefix_gain,
                avatar_cell(node.frame()),
                [y for y in ROW_ANCHORS if int(node.frame()[y][3]) == 8],
            )
    print(
        "PREFIX", prefix_gain, node.levels_completed, node.terminal(),
        avatar_cell(node.frame()), edge_rows(node.frame()),
    )
    cumulative = 0
    for cycle in range(12):
        for token in CYCLE:
            action = gravity_action(node.frame()) if token == "gravity" else token
            if action is None:
                break
            before = arr(node.frame()).copy()
            node.step(*action)
            cumulative += band_shift(before, node.frame())
            if node.terminal() or node.levels_completed > 6:
                break
        print(
            "CYCLE", cycle + 1, cumulative, node.levels_completed,
            node.terminal(), avatar_cell(node.frame()), edge_rows(node.frame()),
        )
        if cycle == 0:
            best = node.clone()
            route = [(3,), click_action(7, 2), "gravity", (4,), "gravity"]
            executed = []
            before = arr(best.frame()).copy()
            for token in route:
                action = (
                    gravity_action(best.frame())
                    if token == "gravity" else token
                )
                if action is None:
                    break
                best.step(*action)
                executed.append(action)
            print(
                "BEST", executed, band_shift(before, best.frame()),
                best.levels_completed, best.terminal(), avatar_cell(best.frame()),
                edge_rows(best.frame()),
            )
            print("BEST_STATE", lattice(best.frame()), shaped_cells(best.frame()))
            break
        if not gravity_action(node.frame()):
            print("SECTION", lattice(node.frame()), shaped_cells(node.frame()))
            for stage in range(20):
                before = arr(node.frame()).copy()
                route, risen, expanded = next_live_rise(node)
                gain = band_shift(before, risen.frame())
                print(
                    "NEXT", stage + 1, route, expanded, gain,
                    risen.levels_completed, risen.terminal(),
                    avatar_cell(risen.frame()),
                )
                if not route:
                    print(
                        "BLOCKED", lattice(node.frame()),
                        shaped_cells(node.frame()), edge_rows(node.frame()),
                    )
                    break
                if risen.terminal() or risen.levels_completed > 6:
                    break
                node = risen
            break
        if node.terminal() or node.levels_completed > 6:
            break


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
