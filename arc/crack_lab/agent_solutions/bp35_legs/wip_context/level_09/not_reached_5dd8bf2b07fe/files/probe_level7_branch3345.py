import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from probe_level7_no_control import (
    PREFIX, SUFFIX, advance, avatar_cell, controls, pair_trials,
)


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


STAGE = (6, 33, 45)


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    root = env.clone()
    gain = advance(root, [*PREFIX, STAGE, *SUFFIX])
    print(
        "ROOT", gain, root.levels_completed, root.terminal(),
        None if root.terminal() else avatar_cell(root.frame()),
        [] if root.terminal() else controls(root.frame()),
        flush=True,
    )
    if root.terminal():
        return
    path, outcomes = pair_trials(root)
    print("OUTCOMES", len(outcomes))
    if path:
        print("WIN", [*PREFIX, STAGE, *SUFFIX, *path], flush=True)
        return
    for outcome in outcomes[:40]:
        print("OPTION", outcome)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
