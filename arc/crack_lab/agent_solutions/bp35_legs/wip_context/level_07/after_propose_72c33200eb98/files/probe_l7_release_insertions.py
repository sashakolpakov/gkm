"""Test one omitted action-7 release at every decoded-route boundary."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena
import gkm_legs as campaign

from probe_level7_coordinate_decode import advance
from probe_level7_decoded_stage import decoded_route
from probe_level7_reward_recovery import controls


if campaign._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def probe(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    route = decoded_route()
    for boundary in range(len(route) + 1):
        candidate = [*route[:boundary], (7,), *route[boundary:]]
        node = env.clone()
        advance(node, [*candidate, (3,), (3,), (3,), (3,)])
        if node.levels_completed > 6:
            print("RELEASE_WIN_BEFORE", boundary, candidate, flush=True)
            return
        if node.terminal():
            continue
        for y in controls(node.frame()):
            child = node.clone()
            child.step(6, 3, y)
            if child.levels_completed > 6:
                print(
                    "RELEASE_WIN",
                    boundary,
                    y,
                    [
                        *candidate,
                        (3,), (3,), (3,), (3,), (6, 3, y),
                    ],
                    flush=True,
                )
                return
        if boundary % 10 == 0:
            print("RELEASE_CHECKED", boundary, flush=True)
    print("NO_RELEASE_WIN", len(route) + 1, flush=True)


levels, replay, error = arena.run_program("bp35", probe)
print("RESULT", levels, len(replay), error)
