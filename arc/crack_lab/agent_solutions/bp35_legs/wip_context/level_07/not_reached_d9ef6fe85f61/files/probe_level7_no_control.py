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

STAGES = [
    (6, 27, 33),
    (6, 39, 33),
    (6, 27, 39),
    (6, 39, 39),
    (6, 27, 45),
    (6, 33, 45),
    (6, 39, 45),
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


def advance(node, actions):
    gained = 0
    for action in actions:
        if node.terminal():
            break
        before = arr(node.frame()).copy()
        node.step(*action)
        if not node.terminal():
            gained += band_shift(before, node.frame())
    return gained


def movement_trials(root):
    patterns = [
        ((3,),) * 12,
        ((4,),) * 12,
        ((3,), (4,)) * 6,
        ((4,), (3,)) * 6,
        ((3,), (3,), (4,), (4,)) * 3,
        ((4,), (4,), (3,), (3,)) * 3,
        ((3,), (3,), (3,), (4,), (4,), (4,)) * 2,
        ((4,), (4,), (4,), (3,), (3,), (3,)) * 2,
    ]
    for pattern in patterns:
        node = root.clone()
        gain = advance(node, pattern)
        print(
            "MOVE", pattern, node.levels_completed, not node.terminal(), gain,
            None if node.terminal() else avatar_cell(node.frame()),
            () if node.terminal() else controls(node.frame()),
        )
        if node.levels_completed > 6:
            return pattern
    return ()


def pair_trials(root):
    cross_patterns = [
        (),
        ((3,),), ((4,),),
        ((3,), (3,)), ((4,), (4,)),
        ((3,), (3,), (3,)), ((4,), (4,), (4,)),
    ]
    post_patterns = [
        (),
        ((3,),), ((4,),),
        ((3,), (3,)), ((4,), (4,)),
        ((3,), (3,), (3,)), ((4,), (4,), (4,)),
    ]
    outcomes = []
    for pre in ((), ((3,),), ((4,),)):
        staged = root.clone()
        pre_gain = advance(staged, pre)
        for y1 in controls(staged.frame()):
            flipped = staged.clone()
            first_gain = pre_gain + advance(flipped, [(6, 3, y1)])
            if flipped.terminal():
                continue
            for cross in cross_patterns:
                middle = flipped.clone()
                middle_gain = first_gain + advance(middle, cross)
                if middle.levels_completed > 6:
                    return (*pre, (6, 3, y1), *cross), outcomes
                if middle.terminal():
                    continue
                for y2 in controls(middle.frame()):
                    for post in post_patterns:
                        child = middle.clone()
                        path = (
                            *pre, (6, 3, y1), *cross, (6, 3, y2), *post,
                        )
                        gain = middle_gain + advance(
                            child, [(6, 3, y2), *post]
                        )
                        if child.levels_completed > 6:
                            return path, outcomes
                        if child.terminal():
                            continue
                        outcomes.append(
                            (
                                gain,
                                len(controls(child.frame())),
                                avatar_cell(child.frame()),
                                controls(child.frame()),
                                path,
                            )
                        )
    outcomes.sort(key=lambda item: (-item[0], -item[1], len(item[4])))
    return (), outcomes


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    root = env.clone()
    prefix_gain = advance(root, PREFIX)
    print("ROOT", prefix_gain, avatar_cell(root.frame()), controls(root.frame()))
    seen_no_control = set()
    for stage in STAGES:
        node = root.clone()
        branch_gain = advance(node, [stage, *SUFFIX])
        print(
            "BRANCH", stage, branch_gain,
            None if node.terminal() else avatar_cell(node.frame()),
            () if node.terminal() else controls(node.frame()),
        )
        if node.terminal():
            continue
        if len(controls(node.frame())) >= 2:
            path, outcomes = pair_trials(node)
            print("PAIRS", stage, len(outcomes), outcomes[:20], flush=True)
            if path:
                print("WIN", [*PREFIX, stage, *SUFFIX, *path], flush=True)
                return
            continue
        if controls(node.frame()):
            continue
        key = arr(node.frame())[:63].tobytes()
        if key in seen_no_control:
            continue
        seen_no_control.add(key)
        path = movement_trials(node)
        if path:
            print("WIN", [*PREFIX, stage, *SUFFIX, *path], flush=True)
            return


if __name__ == "__main__":
    levels, path, err = A.run_program("bp35", probe)
    print("RESULT", levels, len(path), err)
