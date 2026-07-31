import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import COL_ANCHORS, ROW_ANCHORS, band_shift, click_action, run_actions
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
    (3,),
]


def gravity_action(frame):
    return next(
        ((6, 3, y) for y in ROW_ANCHORS if int(frame[y][3]) == 8),
        None,
    )


def avatar_cell(frame):
    ys, xs = (arr(frame) == 9).nonzero()
    if not len(ys):
        return None
    return (
        min(range(10), key=lambda i: abs(ROW_ANCHORS[i] - int(ys.mean()))),
        min(range(8), key=lambda j: abs(COL_ANCHORS[j] - int(xs.mean()))),
    )


def execute(root, template):
    node = root.clone()
    actions = []
    for token in template:
        action = gravity_action(node.frame()) if token == "g" else token
        if action is None or node.terminal():
            break
        node.step(*action)
        actions.append(action)
    return node, actions


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    run_actions(env, PREFIX)
    base = arr(env.frame()).copy()
    print("START", avatar_cell(base))
    found = []
    tested = 0
    for support in (click_action(7, 1), click_action(8, 1), click_action(6, 2)):
        for pre_count in range(3):
            for cross in ((3,), (4,)):
                for cross_count in range(1, 5):
                    for post_count in range(3):
                        template = (
                            [(3,)] * pre_count
                            + [support, "g"]
                            + [cross] * cross_count
                            + ["g"]
                            + [cross] * post_count
                        )
                        node, actions = execute(env, template)
                        tested += 1
                        gain = band_shift(base, node.frame())
                        if node.levels_completed > 6 or (
                            gain > 0 and not node.terminal()
                            and avatar_cell(node.frame()) is not None
                        ):
                            found.append(
                                (
                                    node.levels_completed, gain, len(actions),
                                    actions, avatar_cell(node.frame()),
                                )
                            )
    found.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    print("TESTED", tested)
    for item in found[:20]:
        print("FOUND", item)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
