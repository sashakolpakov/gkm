"""Reuse the gravity-room planner from the decoded high turnaround state."""

import json
import os
import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import gravity_room_search, run_actions
from probe_l7_decode_matrix import build, controls
from probe_level7_reward_recovery import avatar_cell, lattice


FLAGS = (False, False, True, False, True)


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)
    with open("frontier_scaffold.json") as stream:
        raw = json.load(stream)["staged_prefix_actions"]
    base_level = int(env.levels_completed)
    prefix = []
    for action in build(raw, FLAGS):
        candidate = action
        if (
            len(action) == 3
            and action[0] == 6
            and action[1] <= 5
            and int(env.frame()[action[2]][action[1]]) != 8
        ):
            visible = controls(env.frame())
            if visible:
                candidate = visible[0]
        env.step(*candidate)
        prefix.append(candidate)
    # Six releases expose the side-room gravity control.
    releases = [(7,)] * 6
    run_actions(env, releases)
    print(
        "FINAL_GRAVITY_ROOT", avatar_cell(env.frame()),
        controls(env.frame()), lattice(env.frame()), flush=True,
    )
    started = time.monotonic()
    route = gravity_room_search(
        env,
        max_states=int(os.environ.get("MAX_STATES", "350")),
        max_depth=int(os.environ.get("MAX_DEPTH", "60")),
        debug=False,
    )
    print(
        "FINAL_GRAVITY_PLAN", len(route), route,
        round(time.monotonic() - started, 2), flush=True,
    )
    if route:
        verified = env.clone()
        run_actions(verified, route)
        print(
            "FINAL_GRAVITY_VERIFY", int(verified.levels_completed),
            bool(verified.terminal()), avatar_cell(verified.frame()),
            controls(verified.frame()), flush=True,
        )
        if verified.levels_completed > base_level:
            print(
                "FINAL_GRAVITY_WIN",
                [*prefix, *releases, *route], flush=True,
            )


arena.run_program("bp35", probe)
