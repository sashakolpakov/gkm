"""Compare the corrected nine-key level-6 entry with the canonical state."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import arr, safe_step


def click_move(node, source, destination):
    for row, col in (source, destination):
        safe_step(node, (6, col + 1, row + 1))


def observe(env):
    with open("checkpoint.json") as stream:
        path = json.load(stream)["final_path"]
    for action in path[:238]:
        safe_step(env, action)

    canonical = env.clone()
    coordinate_actions = 0
    for action in path[238:331]:
        safe_step(canonical, action)
        if isinstance(action, list):
            coordinate_actions += 1
            if coordinate_actions == 28:
                break

    branch = env.clone()
    # The first eleven coordinate macros are immediate and unchanged.
    coordinate_actions = 0
    for action in path[238:]:
        if isinstance(action, list):
            coordinate_actions += 1
        safe_step(branch, action)
        if coordinate_actions == 22:
            break
    for action in (4, 4, 4, 4, 4, 4, 4, 1, 1):
        safe_step(branch, action)
    click_move(branch, (30, 28), (18, 28))
    click_move(branch, (18, 28), (18, 40))
    for action in (4, 4, 1, 1):
        safe_step(branch, action)
    click_move(branch, (30, 46), (18, 46))

    print("CONVERGENCE", {
        "same": bool((arr(canonical.frame())[1:, :] ==
                      arr(branch.frame())[1:, :]).all()),
        "canonical_level": canonical.levels_completed,
        "branch_level": branch.levels_completed,
    })


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path),
                       "error": str(error)})
