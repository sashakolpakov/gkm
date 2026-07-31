"""Bounded reuse probe for the existing gravity/support search."""

import json
import os
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import gravity_room_search, run_actions


def probe(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    base_level = int(env.levels_completed)
    started = time.monotonic()
    route = gravity_room_search(
        env,
        max_states=int(os.environ.get("PROBE_STATES", "900")),
        max_depth=int(os.environ.get("PROBE_DEPTH", "48")),
        debug=os.environ.get("PROBE_DEBUG") == "1",
    )
    elapsed = time.monotonic() - started
    clone = env.clone()
    if route:
        run_actions(clone, route)
    print(
        {
            "route_len": len(route),
            "route": route,
            "level_delta": int(clone.levels_completed) - base_level,
            "terminal": bool(clone.terminal()),
            "seconds": round(elapsed, 3),
        }
    )


arena.run_program("bp35", probe)
