import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import (
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

FIRST_MACROS = [
    [(6, 3, 3), (3,), (6, 3, 57)],
    [(6, 3, 15), (3,), (6, 3, 57)],
    [(6, 3, 51), (3,), (6, 3, 57)],
]

BASE_ROUTE = [click_action(2, 2), click_action(4, 4), *TAIL]


def avatar_cell(frame):
    ys, xs = (arr(frame) == 9).nonzero()
    if not len(ys):
        return None
    return (
        min(range(10), key=lambda i: abs(3 + 6 * i - int(ys.mean()))),
        min(range(8), key=lambda j: abs(15 + 6 * j - int(xs.mean()))),
    )


def controls(frame):
    return [y for y in ROW_ANCHORS if int(frame[y][3]) == 8]


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
        and _cell_shape(frame, i, j)[1] < 21
    ]


def advance(node, actions):
    gain = 0
    for action in actions:
        if node.terminal():
            break
        before = arr(node.frame()).copy()
        node.step(*action)
        if not node.terminal():
            gain += band_shift(before, node.frame())
    return gain


def replay(root, route, score_start):
    node = root.clone()
    gain = 0
    for index, action in enumerate(route):
        if node.terminal():
            break
        before = arr(node.frame()).copy() if index >= score_start else None
        node.step(*action)
        if before is not None and not node.terminal():
            gain += band_shift(before, node.frame())
    return node, gain


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    root, _ = replay(env, BASE_ROUTE, len(BASE_ROUTE))
    print(
        "ROOT",
        avatar_cell(root.frame()),
        controls(root.frame()),
        supports(root.frame()),
        flush=True,
    )

    unique = {}
    tested = 0
    for first_route in FIRST_MACROS:
        first_path = [*BASE_ROUTE, *first_route]
        first, first_gain = replay(env, first_path, len(BASE_ROUTE))
        for support in (None, *supports(first.frame())):
            support_route = [] if support is None else [support]
            supported_path = [*first_path, *support_route]
            supported, _ = replay(env, supported_path, len(BASE_ROUTE))
            for y1 in controls(supported.frame()):
                for cross in ((3,), (4,)):
                    middle_path = [
                        *supported_path,
                        (6, 3, y1),
                        cross,
                    ]
                    middle, _ = replay(env, middle_path, len(BASE_ROUTE))
                    if middle.terminal():
                        continue
                    for y2 in controls(middle.frame()):
                        route = [*middle_path, (6, 3, y2)]
                        child, gain = replay(env, route, len(BASE_ROUTE))
                        tested += 1
                        if child.levels_completed > 6:
                            print("WIN", route, flush=True)
                            return
                        if child.terminal():
                            continue
                        frame = arr(child.frame())
                        key = (
                            frame[:63].tobytes(),
                            moves_used(frame) % 2,
                        )
                        current = unique.get(key)
                        if current is None or gain > current[0]:
                            unique[key] = (
                                gain,
                                route,
                                avatar_cell(frame),
                                controls(frame),
                            )

    outcomes = []
    for gain, route, avatar, remaining in unique.values():
        best_move = (0, None, avatar, remaining)
        for move in ((3,), (4,)):
            child, moved_gain = replay(
                env, [*route, move], len(BASE_ROUTE)
            )
            extra = moved_gain - gain
            if child.levels_completed > 6:
                print("WIN", [*route, move], flush=True)
                return
            if (
                not child.terminal()
                and (extra, len(controls(child.frame()))) >
                    (best_move[0], len(best_move[3]))
            ):
                best_move = (
                    extra,
                    move,
                    avatar_cell(child.frame()),
                    controls(child.frame()),
                )
        outcomes.append(
            (
                gain + best_move[0],
                gain,
                best_move,
                avatar,
                remaining,
                route[len(BASE_ROUTE):],
            )
        )
    outcomes.sort(key=lambda item: (-len(item[4]), -item[0], item[3]))
    print("TESTED", tested, "UNIQUE", len(unique))
    for outcome in outcomes[:30]:
        print("LANDING", outcome)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
