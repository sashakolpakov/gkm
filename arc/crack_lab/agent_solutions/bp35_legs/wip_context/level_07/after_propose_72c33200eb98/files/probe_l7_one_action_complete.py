"""Exhaust the single omitted action in the decoded 60-move witness."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena
import gkm_legs as campaign

from legs import (
    _cell_shape,
    click_action,
)
from probe_level7_decoded_stage import decoded_route
from probe_level7_reward_recovery import avatar_cell, controls


if campaign._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


LEFT, RIGHT, UNDO = (3,), (4,), (7,)
NOOP = (6, 9, 3)
BASE = [*decoded_route(), LEFT, LEFT, LEFT, LEFT]


def visible_actions(frame):
    mode = os.environ.get("ACTION_MODE", "objects")
    actions = []
    if mode == "generic":
        actions.extend([LEFT, RIGHT, UNDO, NOOP, (6, 3, 3)])
    for i in range(10):
        for j in range(8):
            color, _area = _cell_shape(frame, i, j)
            if mode == "objects" and color in (8, 12, 14, 15):
                actions.append(click_action(i, j))
    return list(dict.fromkeys(actions))


def advance(node, actions):
    for action in actions:
        if node.terminal():
            break
        node.step(*action)
        if node.levels_completed > 6:
            break


def finish(node):
    if node.levels_completed > 6:
        return None
    if node.terminal():
        return None
    visible = controls(node.frame())
    if not visible:
        return None
    action = (6, 3, max(visible))
    node.step(*action)
    return action


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)

    root = env.clone()
    tracker = root.clone()
    shard = int(os.environ.get("SHARD", "0"))
    shards = int(os.environ.get("SHARDS", "1"))
    tested = 0
    outcomes = []

    for boundary in range(len(BASE) + 1):
        if tracker.terminal():
            break
        if boundary % shards != shard:
            if boundary < len(BASE):
                tracker.step(*BASE[boundary])
            continue
        actions = visible_actions(tracker.frame())
        for inserted in actions:
            tested += 1
            node = tracker.clone()
            advance(node, [inserted, *BASE[boundary:]])
            final = finish(node)
            route = [*BASE[:boundary], inserted, *BASE[boundary:]]
            if node.levels_completed > 6:
                if final is not None:
                    route.append(final)
                print(
                    "ONE_ACTION_WIN", boundary, inserted, len(route),
                    route, flush=True,
                )
                return
            if not node.terminal():
                outcomes.append(
                    (
                        len(controls(node.frame())),
                        avatar_cell(node.frame()),
                        boundary,
                        inserted,
                        final,
                    )
                )
            if tested % 100 == 0:
                print(
                    "ONE_ACTION_PROGRESS", shard, tested, boundary,
                    flush=True,
                )

        if boundary < len(BASE):
            tracker.step(*BASE[boundary])
            if tracker.terminal():
                break

    outcomes.sort(
        key=lambda item: (
            -item[0],
            item[1] is None,
            item[1] or (99, 99),
        )
    )
    print("ONE_ACTION_NONE", shard, tested, flush=True)
    for outcome in outcomes[:40]:
        print("ONE_ACTION_OPTION", outcome, flush=True)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
