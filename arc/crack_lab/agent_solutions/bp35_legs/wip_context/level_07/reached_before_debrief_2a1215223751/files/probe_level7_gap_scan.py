import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import ROW_ANCHORS, band_shift
from perception import arr
from probe_level7_reward_recovery import avatar_cell, controls


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


NOOP = (6, 9, 3)


def shifted_scaffold():
    with open("frontier_scaffold.json") as stream:
        raw = json.load(stream)["staged_prefix_actions"]
    route = []
    for item in raw:
        action = (item,) if isinstance(item, int) else tuple(item)
        if len(action) == 3 and action[1] != 3:
            action = (action[0], action[1] + 12, action[2])
        route.append(action)
    return route


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

    route = shifted_scaffold()
    outcomes = []
    for insertion in range(len(route) + 1):
        node = env.clone()
        candidate = [*route[:insertion], NOOP, *route[insertion:]]
        height = advance(node, [*candidate, (3,), (3,), (3,), (3,)])
        if node.levels_completed > 6:
            print("WIN_BEFORE_SWITCH", insertion, candidate, flush=True)
            return
        if node.terminal():
            outcomes.append((False, height, insertion, None, ()))
            continue
        visible = [
            y for y in ROW_ANCHORS if int(node.frame()[y][3]) == 8
        ]
        for y in visible:
            child = node.clone()
            child.step(6, 3, y)
            if child.levels_completed > 6:
                print(
                    "GAP_WIN", insertion, y,
                    [*candidate, (3,), (3,), (3,), (3,), (6, 3, y)],
                    flush=True,
                )
                return
        outcomes.append(
            (
                True, height, insertion, avatar_cell(node.frame()),
                tuple(controls(node.frame())),
            )
        )
    outcomes.sort(
        key=lambda item: (-item[0], -item[1], -len(item[4]), item[2])
    )
    print("NO_GAP_WIN", len(outcomes))
    for outcome in outcomes[:30]:
        print("GAP", outcome)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
