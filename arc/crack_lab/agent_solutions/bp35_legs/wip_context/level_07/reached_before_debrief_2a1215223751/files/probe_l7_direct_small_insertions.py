import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import _cell_shape, click_action
from perception import arr
from probe_level7_decoded_stage import decoded_route
from probe_level7_reward_recovery import avatar_cell, controls, lattice


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


BASE = [*decoded_route(), (3,), (3,), (3,), (3,)]
SELECTED = int(os.environ.get("L7_CANDIDATE_INDEX", "-1"))
CENTRAL_X = (27, 33, 39)
observation = {"selected": None, "control_deltas": []}


def replay_checkpoint(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)


def small_supports(frame):
    return [
        click_action(i, j)
        for i in range(10)
        for j in range(8)
        if click_action(i, j)[1] in CENTRAL_X
        and _cell_shape(frame, i, j)[0] in (12, 14)
        and _cell_shape(frame, i, j)[1] < 21
    ]


def program(env):
    replay_checkpoint(env)
    candidate_index = 0
    counts = []
    for boundary in range(len(BASE) + 1):
        actions = small_supports(env.frame())
        counts.append(len(actions))
        for action in actions:
            if candidate_index == SELECTED:
                observation["selected"] = (boundary, action)
                env.step(*action)
                if env.levels_completed > 6:
                    return
                break
            candidate_index += 1
        else:
            if boundary < len(BASE):
                before = arr(env.frame()).copy()
                env.step(*BASE[boundary])
                if boundary in (46, 48):
                    observation["control_deltas"].append(
                        int((before[:63] != arr(env.frame())[:63]).sum())
                    )
                if env.levels_completed > 6:
                    return
            continue

        # The selected insertion has been made; finish without enumerating
        # the now-divergent later frames.
        for index in range(boundary, len(BASE)):
            before = arr(env.frame()).copy()
            env.step(*BASE[index])
            if index in (46, 48):
                observation["control_deltas"].append(
                    int((before[:63] != arr(env.frame())[:63]).sum())
                )
            if env.levels_completed > 6:
                return
        break
    observation["candidate_count"] = candidate_index
    observation["counts"] = counts
    available = controls(env.frame())
    observation["before"] = (
        avatar_cell(env.frame()),
        tuple(available),
        lattice(env.frame()),
    )
    if available:
        env.step(6, 3, max(available))
    observation["after"] = (
        avatar_cell(env.frame()),
        tuple(controls(env.frame())),
        lattice(env.frame()),
    )


levels, path, err = A.run_program("bp35", program)
if levels > 6:
    print(
        "DIRECT_SMALL_WIN", observation["selected"],
        "PATH", path, "OBS", observation, flush=True,
    )
else:
    before = observation.get("before")
    after = observation.get("after")
    print(
        "DIRECT_SMALL_RESULT", SELECTED, levels, len(path), err,
        "selected", observation.get("selected"),
        "control_deltas", observation.get("control_deltas"),
        "before", None if before is None else before[:2],
        "after", None if after is None else after[:2],
        "lattice", None if after is None else after[2],
        "count", observation.get("candidate_count"),
        flush=True,
    )
