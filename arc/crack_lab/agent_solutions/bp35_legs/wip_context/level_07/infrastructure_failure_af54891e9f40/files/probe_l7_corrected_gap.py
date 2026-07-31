"""Test one inert timing action at boundaries of the corrected staged route."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_level7_decoded_stage import decoded_route
from probe_level7_reward_recovery import avatar_cell, controls, lattice


LEFT = (3,)
NOOP = (6, 9, 3)


def run_boundary(boundary):
    result = {}

    def probe(env):
        with open("checkpoint.json") as stream:
            for action in json.load(stream)["final_path"]:
                env.step(action)
        base_level = int(env.levels_completed)
        route = decoded_route()
        candidate = [
            *route[:boundary], NOOP, *route[boundary:],
            LEFT, LEFT, LEFT, LEFT,
        ]
        for action in candidate:
            env.step(*action)
            if env.terminal() or env.levels_completed > base_level:
                break
        final = None
        if not env.terminal() and env.levels_completed == base_level:
            visible = controls(env.frame())
            if visible:
                final = (6, 3, max(visible))
                env.step(*final)
        result.update(
            level=int(env.levels_completed),
            terminal=bool(env.terminal()),
            avatar=None if env.terminal() else avatar_cell(env.frame()),
            controls=() if env.terminal() else tuple(controls(env.frame())),
            final=final,
            lattice="" if env.terminal() else lattice(env.frame()),
            candidate=candidate,
        )

    levels, path, error = arena.run_program("bp35", probe)
    print(
        "CORRECTED_GAP", boundary,
        result["level"], result["terminal"], result["avatar"],
        result["controls"], result["final"],
        "runner", (levels, len(path), error),
        flush=True,
    )
    if result["level"] > 6:
        print("CORRECTED_GAP_WIN", result["candidate"], result["final"], flush=True)
        return True
    return False


start = int(os.environ.get("START", "0"))
stop = int(os.environ.get("STOP", str(len(decoded_route()) + 1)))
for insertion in range(start, stop):
    if run_boundary(insertion):
        break
