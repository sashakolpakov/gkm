import json
import os
import sys
from collections import Counter, deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import (
    COL_ANCHORS,
    ROW_ANCHORS,
    _cell_shape,
    band_shift,
    click_action,
)
from perception import arr


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


TAIL = [
    (4,), (4,), (4,), click_action(8, 4), (6, 3, 3),
    (4,), (6, 3, 3), (4,),
    (3,), (3,), click_action(8, 3), (6, 3, 3), (3,), (6, 3, 9),
]

CONTINUATION = [
    (6, 33, 39),
    click_action(5, 2),
    (6, 33, 57), (6, 3, 33), (4,), (6, 3, 45),
    (4,),
    (6, 39, 33), (6, 3, 9), (3,), (6, 3, 51),
    (6, 33, 33),
    (6, 33, 33),
]


def avatar_cell(frame):
    ys, xs = (arr(frame) == 9).nonzero()
    if not len(ys):
        return None
    return (
        min(range(10), key=lambda i: abs(ROW_ANCHORS[i] - int(ys.mean()))),
        min(range(8), key=lambda j: abs(COL_ANCHORS[j] - int(xs.mean()))),
    )


def controls(frame):
    return tuple(y for y in ROW_ANCHORS if int(frame[y][3]) == 8)


def support_shapes(frame):
    return tuple(
        (i, j, _cell_shape(frame, i, j)[1])
        for i in range(10)
        for j in range(8)
        if _cell_shape(frame, i, j)[0] in (12, 14)
    )


def lattice(frame):
    palette = {
        0: "v", 3: "#", 5: "#", 8: "g", 9: "A", 10: ".",
        11: "a", 12: "s", 14: "Y", 15: "h",
    }
    return "/".join(
        "".join(palette.get(int(frame[y][x]), "?") for x in COL_ANCHORS)
        for y in ROW_ANCHORS
    )


def step(node, action):
    before = arr(node.frame()).copy()
    node.step(*action)
    return 0 if node.terminal() else band_shift(before, node.frame())


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    root = env.clone()
    route = [
        click_action(2, 2),
        click_action(4, 2),
        click_action(4, 4),
        click_action(1, 3),
        *TAIL,
        (6, 39, 27),
        (3,),
        *CONTINUATION[:6],
        (4,), (4,),
        (3,), (6, 3, 9), (3,), (6, 3, 51),
        (3,), (6, 3, 15), (4,), (6, 3, 51),
        (3,), (6, 3, 21), (4,), (6, 3, 57),
        (4,), (6, 3, 57), (3,), (6, 3, 15),
        (3,), (6, 3, 27), (4,), (6, 3, 39),
        (3,),
    ]
    route = route[:-9]
    height = 0
    for action in route:
        height += step(root, action)
    print(
        "ROOT", height, avatar_cell(root.frame()), controls(root.frame()),
        support_shapes(root.frame()), lattice(root.frame()),
    )

    if not controls(root.frame()):
        if height >= 52:
            return

        def local_actions(frame):
            ai, aj = avatar_cell(frame)
            actions = [(3,), (4,)]
            actions.extend(
                click_action(i, j)
                for i in range(max(0, ai - 2), min(10, ai + 3))
                for j in range(max(0, aj - 2), min(8, aj + 3))
                if _cell_shape(frame, i, j)[0] in (12, 14)
            )
            return actions

        queue = deque([(root.clone(), (), 0)])
        seen = {(arr(root.frame())[:63].tobytes(), 0)}
        expanded = 0
        best = (0, ())
        while queue and expanded < 300:
            node, path, gained = queue.popleft()
            if len(path) >= 7:
                continue
            for action in local_actions(node.frame()):
                child = node.clone()
                total = gained + step(child, action)
                expanded += 1
                child_path = (*path, action)
                if child.levels_completed > 6:
                    print("LOCAL_WIN", expanded, child_path)
                    return
                if child.terminal():
                    continue
                if total > best[0]:
                    best = (total, child_path)
                    print(
                        "LOCAL_PROGRESS", expanded, total,
                        avatar_cell(child.frame()), child_path,
                    )
                key = (
                    arr(child.frame())[:63].tobytes(),
                    len(child_path) % 2,
                )
                if key not in seen:
                    seen.add(key)
                    queue.append((child, child_path, total))
                if expanded >= 300:
                    break
        print("LOCAL_SEARCH", expanded, len(seen), best)
        return

    if height == 35:
        candidates = (
            click_action(5, 2),
            click_action(5, 4),
            click_action(6, 2),
            click_action(6, 4),
            click_action(7, 2),
            click_action(7, 3),
            click_action(7, 4),
        )
        suffix = (
            (4,), (6, 3, 57), (3,), (6, 3, 15),
            (3,), (6, 3, 27), (4,), (6, 3, 39),
        )
        for support in candidates:
            staged = root.clone()
            gained = step(staged, support)
            for action in suffix:
                if staged.terminal():
                    break
                gained += step(staged, action)
            print(
                "STAGED_END", support, gained, staged.levels_completed,
                not staged.terminal(),
                None if staged.terminal() else avatar_cell(staged.frame()),
                () if staged.terminal() else controls(staged.frame()),
                "" if staged.terminal() else lattice(staged.frame()),
            )
            if staged.terminal():
                continue
            for movement in ((3,), (4,)):
                child = staged.clone()
                move_gain = step(child, movement)
                print(
                    "STAGED_MOVE", support, movement, move_gain,
                    child.levels_completed, not child.terminal(),
                    None if child.terminal() else avatar_cell(child.frame()),
                )
        return

    representatives = {}
    counts = Counter()
    for support in (
        click_action(5, 2),
        click_action(5, 4),
        click_action(6, 2),
        click_action(6, 4),
        click_action(7, 2),
        click_action(7, 3),
        click_action(7, 4),
    ):
        supported = root.clone()
        support_gain = 0 if support is None else step(supported, support)
        if supported.terminal():
            continue
        for normal in ((4,),):
            staged = supported.clone()
            gained0 = support_gain if normal is None else support_gain + step(staged, normal)
            if staged.terminal():
                continue
            for y1 in controls(staged.frame()):
                flipped = staged.clone()
                gained1 = gained0 + step(flipped, (6, 3, y1))
                if flipped.levels_completed > 6:
                    print("WIN", (support, normal, (6, 3, y1)))
                    return
                if flipped.terminal():
                    continue
                for cross in ((3,), (4,)):
                    middle = flipped.clone()
                    gained2 = gained1 + step(middle, cross)
                    if middle.levels_completed > 6:
                        print("WIN", (support, normal, (6, 3, y1), cross))
                        return
                    if middle.terminal():
                        continue
                    for y2 in controls(middle.frame()):
                        child = middle.clone()
                        gained = gained2 + step(child, (6, 3, y2))
                        path = tuple(
                            action for action in (
                                support, normal, (6, 3, y1), cross,
                                (6, 3, y2),
                            )
                            if action is not None
                        )
                        if child.levels_completed > 6:
                            print("WIN", path)
                            return
                        if child.terminal():
                            continue
                        frame = child.frame()
                        signature = (
                            gained,
                            avatar_cell(frame),
                            controls(frame),
                            support_shapes(frame),
                        )
                        counts[signature] += 1
                        representatives.setdefault(signature, path)

    ordered = sorted(
        representatives,
        key=lambda item: (-item[0], -len(item[2]), item[1]),
    )
    print("UNIQUE", len(ordered))
    for signature in ordered[:20]:
        print("OPTION", counts[signature], signature, representatives[signature])


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
