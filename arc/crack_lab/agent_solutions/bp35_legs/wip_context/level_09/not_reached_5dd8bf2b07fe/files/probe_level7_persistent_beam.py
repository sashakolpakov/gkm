import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import (
    ROW_ANCHORS, _cell_shape, band_shift, click_action, moves_used,
    run_actions,
)
from perception import arr


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


PREFIX = [
    click_action(2, 2),
    (4,), (4,), (4,), click_action(8, 4), (6, 3, 3),
    (4,), (6, 3, 3), (4,),
    (3,), (3,), click_action(8, 3), (6, 3, 3), (3,), (6, 3, 9),
    (3,), (4,), (4,), click_action(8, 2), (6, 3, 9),
    (3,), (3,), (6, 3, 15), (3,), (3,),
    (4,), click_action(7, 2), (6, 3, 21), (4,), (6, 3, 15),
    (6, 3, 27),
]


def avatar_cell(frame):
    ys, xs = (arr(frame) == 9).nonzero()
    if not len(ys):
        return None
    return (
        min(range(10), key=lambda i: abs(ROW_ANCHORS[i] - int(ys.mean()))),
        min(range(8), key=lambda j: abs(15 + 6 * j - int(xs.mean()))),
    )


def controls(frame):
    return tuple(y for y in ROW_ANCHORS if int(frame[y][3]) == 8)


def support_actions(frame):
    avatar = avatar_cell(frame)
    if avatar is None:
        return ()
    ai, aj = avatar
    return tuple(
        click_action(i, j)
        for i in range(max(0, ai - 1), min(10, ai + 2))
        for j in range(max(0, aj - 1), min(8, aj + 2))
        if _cell_shape(frame, i, j)[0] in (12, 14)
        and _cell_shape(frame, i, j)[1] < 21
    )


def expanded_count(frame):
    return sum(
        1
        for i in range(10)
        for j in range(8)
        if _cell_shape(frame, i, j)[0] in (12, 14)
        and _cell_shape(frame, i, j)[1] >= 21
    )


def step(node, action):
    before = arr(node.frame()).copy()
    node.step(*action)
    return 0 if node.terminal() else band_shift(before, node.frame())


def persistent_signature(path):
    return tuple(
        (index, action)
        for index, action in enumerate(path)
        if action[0] == 6 and action[1] != 3
    )


def macro_successors(root):
    outcomes = {}
    movement_pairs = (
        ((), (3,)),
        ((), (4,)),
        (((3,),), (4,)),
        (((4,),), (3,)),
    )
    for support in (None, *support_actions(root.frame())):
        supported = root if support is None else root.clone()
        support_path = () if support is None else (support,)
        support_gain = 0 if support is None else step(supported, support)
        if supported.levels_completed > 6:
            return [], support_path
        if supported.terminal():
            continue
        for normal_path, cross in movement_pairs:
            staged = supported
            normal_gain = support_gain
            if normal_path:
                staged = supported.clone()
                normal_gain += step(staged, normal_path[0])
            if staged.levels_completed > 6:
                return [], (*support_path, *normal_path)
            if staged.terminal():
                continue
            for y1 in controls(staged.frame()):
                flipped = staged.clone()
                gain1 = normal_gain + step(flipped, (6, 3, y1))
                path1 = (
                    *support_path, *normal_path, (6, 3, y1), cross,
                )
                if flipped.levels_completed > 6:
                    return [], path1[:-1]
                if flipped.terminal():
                    continue
                middle = flipped.clone()
                gain2 = gain1 + step(middle, cross)
                if middle.levels_completed > 6:
                    return [], path1
                if middle.terminal():
                    continue
                for y2 in controls(middle.frame()):
                    child = middle.clone()
                    gain = gain2 + step(child, (6, 3, y2))
                    path = (*path1, (6, 3, y2))
                    if child.levels_completed > 6:
                        return [], path
                    avatar = avatar_cell(child.frame())
                    if child.terminal() or avatar is None or avatar[0] != 6:
                        continue
                    frame = arr(child.frame())
                    key = (
                        frame[:63].tobytes(),
                        moves_used(frame) % 2,
                        support,
                    )
                    current = outcomes.get(key)
                    if current is None or (gain, -len(path)) > (
                        current[0], -len(current[1])
                    ):
                        outcomes[key] = (gain, path, child)
    return list(outcomes.values()), None


def rank(item):
    height, path, node = item
    frame = node.frame()
    avatar = avatar_cell(frame)
    central = min(avatar[1], 7 - avatar[1], 3)
    staged = len(persistent_signature(path))
    return (
        height * 10
        + len(controls(frame)) * 20
        + expanded_count(frame) * 4
        + staged * 2
        + central,
        height,
        len(controls(frame)),
        staged,
        -len(path),
    )


def diverse_beam(items, width):
    ordered = sorted(items, key=rank, reverse=True)
    if not ordered:
        return []
    selectors = (
        rank,
        lambda item: (item[0], rank(item)),
        lambda item: (len(controls(item[2].frame())), rank(item)),
        lambda item: (len(persistent_signature(item[1])), rank(item)),
    )
    beam = []
    for selector in selectors:
        item = max(ordered, key=selector)
        if item not in beam:
            beam.append(item)
    for item in ordered:
        if len(beam) >= width:
            break
        if item not in beam:
            beam.append(item)
    return beam[:width]


def search(root, beam_width=3, max_macros=10):
    beam = [(0, (), root.clone())]
    seen = set()
    generated = 0
    for depth in range(1, max_macros + 1):
        candidates = []
        for height, path, node in beam:
            successors, winning_suffix = macro_successors(node)
            generated += len(successors)
            if winning_suffix:
                return (*path, *winning_suffix), generated
            for gain, suffix, child in successors:
                child_height = height + gain
                child_path = (*path, *suffix)
                frame = arr(child.frame())
                key = (
                    child_height,
                    frame[:63].tobytes(),
                    moves_used(frame) % 2,
                    persistent_signature(child_path),
                )
                if key in seen:
                    continue
                seen.add(key)
                candidates.append((child_height, child_path, child))
        beam = diverse_beam(candidates, beam_width)
        print(
            "BEAM", depth, generated,
            [
                (
                    height, len(path), avatar_cell(node.frame()),
                    controls(node.frame()), len(persistent_signature(path)),
                    path,
                )
                for height, path, node in beam
            ],
            flush=True,
        )
        if not beam:
            break
    return (), generated


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    root = env.clone()
    run_actions(root, PREFIX)
    print(
        "ROOT", avatar_cell(root.frame()), controls(root.frame()),
        support_actions(root.frame()), flush=True,
    )
    route, generated = search(root)
    print("SEARCH", generated, len(route), route)
    if route:
        verified = root.clone()
        run_actions(verified, route)
        print(
            "VERIFY", verified.levels_completed, verified.terminal(),
            avatar_cell(verified.frame()), controls(verified.frame()),
        )
        print("WIN", [*PREFIX, *route], flush=True)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
