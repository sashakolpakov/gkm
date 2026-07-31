import json
import os
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import (
    COL_ANCHORS,
    ROW_ANCHORS,
    _cell_shape,
    band_shift,
    click_action,
    moves_used,
)
from perception import arr


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


TAIL = [
    (4,), (4,), (4,), click_action(8, 4), (6, 3, 3),
    (4,), (6, 3, 3), (4,),
    (3,), (3,), click_action(8, 3), (6, 3, 3), (3,), (6, 3, 9),
    (3,), (4,), (4,), click_action(8, 2), (6, 3, 9),
    (3,), (3,), (6, 3, 15), (3,), (3,),
    (4,), click_action(7, 2), (6, 3, 21), (4,), (6, 3, 15),
    (6, 3, 27),
    (3,), (6, 3, 33), (4,), (6, 3, 33),
    (6, 3, 15), (3,), (6, 3, 45),
    (6, 3, 57), (4,), (6, 3, 51),
    (3,), (6, 3, 27), (4,), (6, 3, 57),
]

CONTINUATION = [
    (6, 33, 39),
    click_action(5, 2),
    (6, 33, 57), (6, 3, 33), (4,), (6, 3, 45),
    (4,),
    (6, 39, 33), (6, 3, 9), (3,), (6, 3, 51),
    (6, 33, 33),
    (6, 33, 33),
    (6, 3, 27), (4,), (6, 3, 27),
    (6, 3, 51), (4,), (6, 3, 33),
    (6, 3, 15), (3,), (6, 3, 39),
    (6, 3, 21), (3,), (6, 3, 57),
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
    return [y for y in ROW_ANCHORS if int(frame[y][3]) == 8]


def lattice(frame):
    palette = {
        3: "#", 5: "#", 8: "g", 9: "A", 10: ".", 11: "a",
        12: "s", 14: "Y", 15: "h",
    }
    return "/".join(
        "".join(palette.get(int(frame[y][x]), "?") for x in COL_ANCHORS)
        for y in ROW_ANCHORS
    )


def supports(frame):
    avatar = avatar_cell(frame)
    if avatar is None:
        return []
    ai, aj = avatar
    return [
        click_action(i, j)
        for i in range(max(0, ai - 2), min(10, ai + 3))
        for j in range(max(0, aj - 1), min(8, aj + 2))
        if _cell_shape(frame, i, j)[0] in (12, 14)
    ]


def expanded_supports(frame):
    return [
        click_action(i, j)
        for i in range(10)
        for j in range(8)
        if _cell_shape(frame, i, j)[0] in (12, 14)
        and _cell_shape(frame, i, j)[1] >= 21
    ]


def step(node, action):
    before = arr(node.frame()).copy()
    node.step(*action)
    if node.terminal():
        return 0
    return band_shift(before, node.frame())


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
        *TAIL[:14],
        (6, 39, 27),
        *TAIL[14:15],
        *CONTINUATION[:-3],
    ]
    total_height = 0
    for index, action in enumerate(route):
        before = arr(root.frame()).copy()
        root.step(*action)
        gained = band_shift(before, root.frame())
        total_height += gained
        if gained:
            print(
                "RISE", index, action, gained, total_height,
                avatar_cell(root.frame()), controls(root.frame()),
                lattice(root.frame()),
            )
            if avatar_cell(root.frame())[0] == 6:
                for direction in ((3,), (4,)):
                    child = root.clone()
                    previous_avatar = avatar_cell(child.frame())
                    for count in range(1, 5):
                        before_side = arr(child.frame()).copy()
                        child.step(*direction)
                        live = not child.terminal()
                        side_gain = (
                            0 if not live
                            else band_shift(before_side, child.frame())
                        )
                        avatar = None if not live else avatar_cell(child.frame())
                        if (
                            child.levels_completed > 6
                            or not live
                            or side_gain
                            or avatar != previous_avatar
                        ):
                            print(
                                "LANDING_MOVE", index, direction, count,
                                child.levels_completed, live, side_gain, avatar,
                                [] if not live else controls(child.frame()),
                            )
                        if not live:
                            break
                        previous_avatar = avatar
            if avatar_cell(root.frame())[0] != 6:
                unknown = [
                    (i, j, int(root.frame()[y][x]))
                    for i, y in enumerate(ROW_ANCHORS)
                    for j, x in enumerate(COL_ANCHORS)
                    if int(root.frame()[y][x]) not in (3, 5, 8, 9, 10, 11, 12, 14, 15)
                ]
                print("UNKNOWN", index, unknown)
                print(
                    "HIGH_SUPPORTS", index,
                    [
                        (action, _cell_shape(root.frame(),
                                             ROW_ANCHORS.index(action[2]),
                                             COL_ANCHORS.index(action[1])))
                        for action in supports(root.frame())
                    ],
                )
                for support in (None, *supports(root.frame())):
                    staged = root.clone()
                    stage_gain = 0 if support is None else step(staged, support)
                    if staged.terminal():
                        print("STAGE_DEAD", index, support, stage_gain)
                        continue
                    for direction in ((3,), (4,)):
                        child = staged.clone()
                        previous_avatar = avatar_cell(child.frame())
                        for count in range(1, 5):
                            before_side = arr(child.frame()).copy()
                            child.step(*direction)
                            live = not child.terminal()
                            side_gain = (
                                0 if not live
                                else band_shift(before_side, child.frame())
                            )
                            avatar = None if not live else avatar_cell(child.frame())
                            if (
                                child.levels_completed > 6
                                or not live
                                or side_gain
                                or avatar != previous_avatar
                            ):
                                print(
                                    "SIDE", index, support, direction, count,
                                    child.levels_completed, live,
                                    stage_gain + side_gain, avatar,
                                    [] if not live else controls(child.frame()),
                                )
                            if not live:
                                break
                            previous_avatar = avatar
    print(
        "ROOT",
        "height", total_height,
        root.levels_completed,
        root.terminal(),
        avatar_cell(root.frame()),
        controls(root.frame()),
        supports(root.frame()),
        expanded_supports(root.frame()),
        lattice(root.frame()),
    )

    for action in [(3,), (4,), *supports(root.frame())]:
        child = root.clone()
        gain = step(child, action)
        print(
            "LOCAL",
            action,
            child.levels_completed,
            not child.terminal(),
            gain,
            None if child.terminal() else avatar_cell(child.frame()),
            [] if child.terminal() else controls(child.frame()),
        )

    if not controls(root.frame()):
        base = arr(root.frame()).copy()
        queue = deque([(root.clone(), [])])
        seen = {(base[:63].tobytes(), moves_used(base) % 2)}
        found = None
        expanded = 0
        while queue and expanded < 120:
            node, path = queue.popleft()
            if len(path) >= 6:
                continue
            for action in [(3,), (4,), *supports(node.frame())]:
                child = node.clone()
                child.step(*action)
                expanded += 1
                child_path = [*path, action]
                if child.levels_completed > 6:
                    found = (child_path, "WIN")
                    queue.clear()
                    break
                if child.terminal():
                    continue
                gain = band_shift(base, child.frame())
                if gain:
                    found = (
                        child_path,
                        gain,
                        avatar_cell(child.frame()),
                        controls(child.frame()),
                    )
                    queue.clear()
                    break
                frame = arr(child.frame())
                key = (frame[:63].tobytes(), moves_used(frame) % 2)
                if key not in seen:
                    seen.add(key)
                    queue.append((child, child_path))
                if expanded >= 120:
                    break
        print("NEXT_RISE", expanded, found)

    outcomes = []
    for y1 in controls(root.frame()):
        flipped = root.clone()
        gain1 = step(flipped, (6, 3, y1))
        queue = deque([(flipped, (), gain1)])
        seen = {arr(flipped.frame())[:63].tobytes()}
        while queue:
            node, movement, gained = queue.popleft()
            for y2 in controls(node.frame()):
                child = node.clone()
                total = gained + step(child, (6, 3, y2))
                if (
                    child.levels_completed > 6
                    or total > 5
                    or (not child.terminal() and avatar_cell(child.frame())[1] != 3)
                ):
                    outcomes.append((
                        child.levels_completed,
                        not child.terminal(),
                        total,
                        None if child.terminal() else avatar_cell(child.frame()),
                        [] if child.terminal() else controls(child.frame()),
                        ((6, 3, y1), *movement, (6, 3, y2)),
                    ))
            if len(movement) >= 7:
                continue
            for action in ((3,), (4,)):
                child = node.clone()
                total = gained + step(child, action)
                if child.levels_completed > 6:
                    outcomes.append((
                        child.levels_completed, False, total, None, [],
                        ((6, 3, y1), *movement, action),
                    ))
                    continue
                if child.terminal():
                    continue
                key = arr(child.frame())[:63].tobytes()
                if key not in seen:
                    seen.add(key)
                    queue.append((child, (*movement, action), total))
    outcomes.sort(key=lambda item: (-item[0], -item[2], len(item[-1])))
    print("CROSS_STATES", len(seen), "OUTCOMES", len(outcomes))
    for outcome in outcomes[:32]:
        print("CROSS", outcome)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
