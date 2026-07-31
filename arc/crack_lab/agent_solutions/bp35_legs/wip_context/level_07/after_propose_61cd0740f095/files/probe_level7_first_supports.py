import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import ROW_ANCHORS, _cell_shape, band_shift, click_action
from perception import arr


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


BASE = [
    click_action(2, 2), click_action(4, 4),
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
    ai, aj = avatar_cell(frame)
    return [
        click_action(i, j)
        for i in range(max(0, ai - 2), min(10, ai + 3))
        for j in range(max(0, aj - 1), min(8, aj + 2))
        if _cell_shape(frame, i, j)[0] in (12, 14)
        and _cell_shape(frame, i, j)[1] < 21
    ]


def replay(root, route):
    node = root.clone()
    gain = 0
    for index, action in enumerate(route):
        if node.terminal():
            break
        before = arr(node.frame()).copy() if index >= len(BASE) else None
        node.step(*action)
        if before is not None and not node.terminal():
            gain += band_shift(before, node.frame())
    return node, gain


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    root, _ = replay(env, BASE)
    support_actions = supports(root.frame())
    print(
        "ROOT",
        avatar_cell(root.frame()),
        controls(root.frame()),
        support_actions,
        flush=True,
    )

    outcomes = []
    for support in support_actions:
        staged_route = [*BASE, support]
        staged, _ = replay(env, staged_route)
        for y1 in controls(staged.frame()):
            for cross in ((3,), (4,)):
                middle_route = [*staged_route, (6, 3, y1), cross]
                middle, _ = replay(env, middle_route)
                if middle.terminal():
                    continue
                for y2 in controls(middle.frame()):
                    route = [*middle_route, (6, 3, y2)]
                    child, gain = replay(env, route)
                    if child.levels_completed > 6:
                        print("WIN", route[len(BASE):], flush=True)
                        return
                    if child.terminal():
                        continue
                    outcomes.append(
                        (
                            len(controls(child.frame())),
                            gain,
                            avatar_cell(child.frame()),
                            controls(child.frame()),
                            route[len(BASE):],
                        )
                    )
    outcomes.sort(key=lambda item: (-item[0], -item[1], item[2]))
    print("OUTCOMES", len(outcomes))
    for outcome in outcomes:
        print("FIRST", outcome)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
