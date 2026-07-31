"""Test one omitted visible gravity click at each decoded-route boundary."""

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
    tracker = env.clone()
    tested = 0
    for boundary in range(len(route) + 1):
        visible_all = (
            controls(tracker.frame()) if not tracker.terminal() else []
        )
        visible = [max(visible_all)] if visible_all else []
        for y in visible:
            tested += 1
            candidate = [
                *route[:boundary], (6, 3, y), *route[boundary:]
            ]
            node = env.clone()
            advance(node, [*candidate, (3,), (3,), (3,), (3,)])
            if node.levels_completed > 6:
                print(
                    "CONTROL_WIN_BEFORE", boundary, y, candidate, flush=True
                )
                return
            if node.terminal():
                continue
            for final_y in controls(node.frame()):
                child = node.clone()
                child.step(6, 3, final_y)
                if child.levels_completed > 6:
                    print(
                        "CONTROL_WIN",
                        boundary,
                        y,
                        final_y,
                        [
                            *candidate,
                            (3,), (3,), (3,), (3,),
                            (6, 3, final_y),
                        ],
                        flush=True,
                    )
                    return
        if boundary < len(route) and not tracker.terminal():
            tracker.step(*route[boundary])
        if boundary % 10 == 0:
            print("CONTROL_CHECKED", boundary, tested, flush=True)
    print("NO_CONTROL_WIN", tested, flush=True)


levels, replay, error = arena.run_program("bp35", probe)
print("RESULT", levels, len(replay), error)
