import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import _cell_shape, band_shift, click_action
from perception import arr
from probe_level7_coordinate_decode import advance
from probe_level7_decoded_stage import decoded_route
from probe_level7_reward_recovery import avatar_cell, controls


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def compact_nearby(frame):
    avatar = avatar_cell(frame)
    if avatar is None:
        return []
    ai, aj = avatar
    return [
        click_action(i, j)
        for i in range(max(0, ai - 2), min(10, ai + 3))
        for j in range(max(0, aj - 2), min(8, aj + 3))
        if _cell_shape(frame, i, j)[0] in (12, 14)
        and _cell_shape(frame, i, j)[1] >= 21
    ]


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    route = decoded_route()
    tracker = env.clone()
    boundaries = {0}
    frames = {0: arr(tracker.frame()).copy()}
    for index, action in enumerate(route):
        if len(action) == 3 and action[1] == 3:
            boundaries.add(index)
            frames[index] = arr(tracker.frame()).copy()
        before = arr(tracker.frame()).copy()
        tracker.step(*action)
        if tracker.terminal():
            break
        if band_shift(before, tracker.frame()):
            boundaries.add(index + 1)
            frames[index + 1] = arr(tracker.frame()).copy()

    candidates = []
    for boundary in sorted(boundaries):
        for action in compact_nearby(frames[boundary]):
            candidates.append((boundary, action))
    print(
        "CANDIDATES", len(candidates), "BOUNDARIES",
        sorted(boundaries), flush=True,
    )

    outcomes = []
    for candidate_index, (boundary, action) in enumerate(candidates, 1):
        candidate_route = [*route[:boundary], action, *route[boundary:]]
        node = env.clone()
        height = advance(
            node,
            [*candidate_route, (3,), (3,), (3,), (3,)],
        )
        if node.levels_completed > 6:
            print(
                "INSERT_WIN_BEFORE", boundary, action,
                candidate_route, flush=True,
            )
            return
        if node.terminal():
            outcomes.append((False, height, boundary, action, None, ()))
            continue
        for y in controls(node.frame()):
            child = node.clone()
            child.step(6, 3, y)
            if child.levels_completed > 6:
                print(
                    "INSERT_WIN", boundary, action, y,
                    [
                        *candidate_route,
                        (3,), (3,), (3,), (3,), (6, 3, y),
                    ],
                    flush=True,
                )
                return
        outcomes.append(
            (
                True, height, boundary, action,
                avatar_cell(node.frame()), tuple(controls(node.frame())),
            )
        )
        if candidate_index % 40 == 0:
            print("CHECKED", candidate_index, flush=True)
    outcomes.sort(
        key=lambda item: (-item[0], -item[1], -len(item[5]))
    )
    print("NO_INSERT_WIN", len(outcomes))
    for outcome in outcomes[:50]:
        print("OPTION", outcome)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
