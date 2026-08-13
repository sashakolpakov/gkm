"""Test color-15 bridge pieces as coordinate-action sources on level 9."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import connected_components, safe_step
from probe_l9_nonlocal_moves import FIRST_RELAY, board, move


def selected(frame):
    return any(
        blob.area >= 4
        for blob in connected_components(frame, colors=(3,))
    )


def fixed_results(root):
    before = board(root.frame())
    holes, bridges, pegs, fixed, carriers = before
    destinations = holes | bridges | pegs | fixed | carriers
    found = []
    tested = 0
    for source in sorted(fixed):
        for destination in sorted(destinations - {source}):
            child = root.clone()
            move(child, source, destination)
            tested += 1
            after = board(child.frame())
            if (
                not selected(child.frame())
                and (after != before
                     or int(child.levels_completed) > int(root.levels_completed))
            ):
                found.append((source, destination,
                              int(child.levels_completed), after[1:]))
    return tested, tuple(found)


def probe(env):
    with open("checkpoint.json") as stream:
        campaign = json.load(stream)["final_path"]
    for action in campaign:
        safe_step(env, tuple(action) if isinstance(action, list) else action)

    context = os.environ.get("OPT_CONTEXT", "entry")
    if context == "far":
        for source, destination in FIRST_RELAY:
            move(env, source, destination)
        for _ in range(9):
            safe_step(env, 4)
    print("fixed_moves", context, fixed_results(env), flush=True)


arena.run_program("lf52", probe)
