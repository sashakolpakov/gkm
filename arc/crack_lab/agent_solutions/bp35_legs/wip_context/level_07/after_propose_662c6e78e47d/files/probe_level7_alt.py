import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import COL_ANCHORS, ROW_ANCHORS, click_action, run_actions
from perception import arr


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


PREFIX = [
    (4,), (4,), (4,), click_action(8, 4), (6, 3, 3),
    (4,), (6, 3, 3), (4,),
    (3,), (3,), click_action(8, 3), (6, 3, 3), (3,), (6, 3, 9),
    (3,), (4,), (4,), click_action(8, 2), (6, 3, 9),
    (3,), (3,), (6, 3, 15), (3,), (3,),
    (4,), click_action(7, 2), (6, 3, 21), (4,), (6, 3, 15),
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


def result(node):
    return (
        node.levels_completed, node.terminal(), avatar_cell(node.frame()),
        controls(node.frame()),
    )


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    run_actions(env, PREFIX + [(6, 3, 15), (3,), (6, 3, 33)])
    print("STAGE", result(env))
    for y1 in controls(env.frame()):
        for move in (3, 4):
            first = env.clone()
            first.step(6, 3, y1)
            first.step(move)
            for y2 in controls(first.frame()):
                child = first.clone()
                child.step(6, 3, y2)
                print("PAIR", y1, move, y2, result(child))
                for finish in (3, 4):
                    done = child.clone()
                    done.step(finish)
                    print("FINISH", y1, move, y2, finish, result(done))
    run_actions(env, [(6, 3, 51), (3,), (6, 3, 27)])
    print("STAGE2", result(env))
    for y1 in controls(env.frame()):
        for move in (3, 4):
            first = env.clone()
            first.step(6, 3, y1)
            first.step(move)
            for y2 in controls(first.frame()):
                child = first.clone()
                child.step(6, 3, y2)
                print("PAIR2", y1, move, y2, result(child))


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
