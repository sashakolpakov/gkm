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

SEED = [
    (6, 3, 15), (3,), (6, 3, 51),
    (4,), (6, 3, 33), (3,), (6, 3, 45),
    (4,), (6, 3, 27), (3,), (6, 3, 57),
]

SUPPORT_FILTER = set()


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
        for i in range(max(0, ai - 2), min(10, ai + 3))
        for j in range(max(0, aj - 1), min(8, aj + 2))
        if _cell_shape(frame, i, j)[0] in (12, 14)
        and _cell_shape(frame, i, j)[1] < 21
    )


def step(node, action):
    before = arr(node.frame()).copy()
    node.step(*action)
    return 0 if node.terminal() else band_shift(before, node.frame())


def macro_successors(root):
    outcomes = {}
    win = None
    support_options = (
        None,
        *(
            action for action in support_actions(root.frame())
            if action in SUPPORT_FILTER
        ),
    )
    for support in support_options:
        supported = root if support is None else root.clone()
        support_path = () if support is None else (support,)
        support_gain = 0 if support is None else step(supported, support)
        if supported.levels_completed > 6:
            return [], support_path
        if supported.terminal():
            continue
        for normal in (None, (3,), (4,)):
            staged = supported if normal is None else supported.clone()
            normal_path = () if normal is None else (normal,)
            normal_gain = (
                support_gain if normal is None
                else support_gain + step(staged, normal)
            )
            if staged.levels_completed > 6:
                return [], (*support_path, *normal_path)
            if staged.terminal():
                continue
            for y1 in controls(staged.frame()):
                flipped = staged.clone()
                gain1 = normal_gain + step(flipped, (6, 3, y1))
                path1 = (*support_path, *normal_path, (6, 3, y1))
                if flipped.levels_completed > 6:
                    return [], path1
                if flipped.terminal():
                    continue
                for cross in ((3,), (4,)):
                    middle = flipped.clone()
                    gain2 = gain1 + step(middle, cross)
                    path2 = (*path1, cross)
                    if middle.levels_completed > 6:
                        return [], path2
                    if middle.terminal():
                        continue
                    for y2 in controls(middle.frame()):
                        child = middle.clone()
                        gain = gain2 + step(child, (6, 3, y2))
                        path = (*path2, (6, 3, y2))
                        if child.levels_completed > 6:
                            return [], path
                        avatar = avatar_cell(child.frame())
                        if child.terminal() or avatar is None or avatar[0] != 6:
                            continue
                        frame = arr(child.frame()).copy()
                        key = (
                            frame[:63].tobytes(),
                            moves_used(frame) % 2,
                        )
                        current = outcomes.get(key)
                        if current is None or (gain, -len(path)) > (
                            current[0], -len(current[1])
                        ):
                            outcomes[key] = (gain, path, child)
    return list(outcomes.values()), win


def rank(item):
    height, path, node = item
    avatar = avatar_cell(node.frame())
    central = min(avatar[1], 7 - avatar[1], 3)
    return (
        height * 10 + len(controls(node.frame())) * 20 + central,
        height,
        len(controls(node.frame())),
        -len(path),
    )


def search(root, beam_width=20, max_macros=1):
    beam = [(0, (), root.clone())]
    seen = set()
    expanded = 0
    for depth in range(1, max_macros + 1):
        candidates = {}
        for height, path, node in beam:
            successors, winning_suffix = macro_successors(node)
            expanded += len(successors)
            if winning_suffix:
                return (*path, *winning_suffix), expanded
            for gain, suffix, child in successors:
                child_height = height + gain
                child_path = (*path, *suffix)
                frame = arr(child.frame())
                key = (
                    child_height,
                    frame[:63].tobytes(),
                    moves_used(frame) % 2,
                )
                if key in seen:
                    continue
                seen.add(key)
                current = candidates.get(key)
                item = (child_height, child_path, child)
                if current is None or rank(item) > rank(current):
                    candidates[key] = item
        ordered = sorted(candidates.values(), key=rank, reverse=True)
        beam = []
        if ordered:
            representatives = [
                ordered[0],
                max(
                    ordered,
                    key=lambda item: (
                        len(controls(item[2].frame())), item[0], rank(item)
                    ),
                ),
                max(ordered, key=lambda item: (item[0], rank(item))),
            ]
            for item in representatives:
                if item not in beam:
                    beam.append(item)
            for item in ordered:
                if len(beam) >= beam_width:
                    break
                if item not in beam:
                    beam.append(item)
        print(
            "BEAM", depth, expanded,
            [
                (
                    height, len(path), avatar_cell(node.frame()),
                    controls(node.frame()), path,
                )
                for height, path, node in beam
            ],
            flush=True,
        )
        if not beam:
            break
    return (), expanded


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    root = env.clone()
    run_actions(root, [*PREFIX, *SEED])
    print(
        "ROOT", avatar_cell(root.frame()), controls(root.frame()),
        support_actions(root.frame()), flush=True,
    )
    if not controls(root.frame()):
        patterns = [
            ((3,),) * 10,
            ((4,),) * 10,
            ((3,), (4,)) * 5,
            ((4,), (3,)) * 5,
        ]
        for pattern in patterns:
            child = root.clone()
            run_actions(child, pattern)
            print(
                "FINISH", pattern, child.levels_completed, child.terminal(),
                None if child.terminal() else avatar_cell(child.frame()),
            )
        return
    route, expanded = search(root)
    print("SEARCH", expanded, len(route), route)
    if route:
        verified = root.clone()
        run_actions(verified, route)
        print(
            "VERIFY", verified.levels_completed, verified.terminal(),
            avatar_cell(verified.frame()), controls(verified.frame()),
        )
        print("WIN", [*PREFIX, *SEED, *route], flush=True)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
