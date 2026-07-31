import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import (
    COL_ANCHORS, ROW_ANCHORS, _cell_shape, band_shift, click_action,
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
]

PREFIX = [
    click_action(2, 2),
    click_action(4, 2),
    click_action(4, 4),
    click_action(1, 3),
    *TAIL,
    (6, 39, 27),
    (3,),
    *CONTINUATION,
    (4,), (4,),
    (3,), (6, 3, 9), (3,), (6, 3, 51),
    (3,), (6, 3, 15), (4,), (6, 3, 51),
    (3,), (6, 3, 21), (4,), (6, 3, 57),
]

SUFFIX = [
    (4,), (6, 3, 57), (3,), (6, 3, 15),
    (3,), (6, 3, 27), (4,), (6, 3, 39),
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


def lattice(frame):
    palette = {
        0: "v", 3: "#", 5: "#", 8: "g", 9: "A", 10: ".",
        11: "a", 12: "s", 14: "Y", 15: "h",
    }
    return "/".join(
        "".join(palette.get(int(frame[y][x]), "?") for x in COL_ANCHORS)
        for y in ROW_ANCHORS
    )


def advance(node, actions):
    height = 0
    for action in actions:
        if node.terminal():
            break
        before = arr(node.frame()).copy()
        node.step(*action)
        if not node.terminal():
            height += band_shift(before, node.frame())
    return height


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    route = [
        *PREFIX,
        click_action(5, 2),
        *SUFFIX,
        (3,), (6, 3, 9), (4,), (6, 3, 39),
    ]
    root = env.clone()
    height = advance(root, route)
    print(
        "ENDGAME_ROOT", len(route), height, root.levels_completed,
        root.terminal(), avatar_cell(root.frame()), controls(root.frame()),
    )
    for left_count in range(9):
        staged = root.clone()
        advance(staged, [(3,)] * left_count)
        visible = controls(staged.frame())
        for y in visible:
            node = staged.clone()
            node.step(6, 3, y)
            print(
                "TRY", left_count, y, node.levels_completed,
                node.terminal(),
                None if node.terminal() else avatar_cell(node.frame()),
                [] if node.terminal() else controls(node.frame()),
            )
            if node.levels_completed > 6:
                print(
                    "WIN_ROUTE",
                    [*route, *([(3,)] * left_count), (6, 3, y)],
                    flush=True,
                )
                return
            if left_count == 3 and not node.terminal():
                print("SIDE_ROOM", y, lattice(node.frame()))
                frame = node.frame()
                local = [(3,), (4,)]
                local += [
                    click_action(i, j)
                    for i in range(10)
                    for j in range(8)
                    if _cell_shape(frame, i, j)[0] in (12, 14, 15)
                ]
                for action in local:
                    child = node.clone()
                    before = arr(child.frame()).copy()
                    child.step(*action)
                    gain = (
                        0 if child.terminal()
                        else band_shift(before, child.frame())
                    )
                    if (
                        child.levels_completed > 6
                        or gain
                        or (
                            not child.terminal()
                            and (
                                avatar_cell(child.frame()) != avatar_cell(frame)
                                or controls(child.frame())
                            )
                        )
                    ):
                        print(
                            "SIDE_ACTION", y, action, gain,
                            child.levels_completed, child.terminal(),
                            None if child.terminal()
                            else avatar_cell(child.frame()),
                            [] if child.terminal()
                            else controls(child.frame()),
                        )

    crossing_root = env.clone()
    crossing_height = advance(
        crossing_root,
        [*PREFIX, click_action(5, 2), *SUFFIX],
    )
    crossing_cores = [
        [(3,), (6, 3, 9), (4,), (6, 3, 39)],
        [(4,), (6, 3, 15), (3,), (6, 3, 39)],
    ]
    for core in crossing_cores:
        middle = crossing_root.clone()
        core_gain = advance(middle, core)
        for direction in ((3,), (4,)):
            child = middle.clone()
            previous = avatar_cell(child.frame())
            gained = 0
            for count in range(9):
                current = avatar_cell(child.frame())
                if (
                    count == 0
                    or current != previous
                    or controls(child.frame())
                    or child.levels_completed > 6
                ):
                    print(
                        "CROSS_VARIANT", crossing_height + core_gain + gained,
                        core, direction, count, child.levels_completed,
                        child.terminal(), current, controls(child.frame()),
                    )
                if child.levels_completed > 6 or child.terminal():
                    break
                previous = current
                gained += advance(child, [direction])


if __name__ == "__main__":
    levels, path, err = A.run_program("bp35", probe)
    print("RESULT", levels, len(path), err)
