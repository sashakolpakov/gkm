"""Falsify or confirm existing gravity/support legs on pristine level 9."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

import solve
from legs import (
    cross_persistent_support_rooms,
    cross_staged_gravity_zigzag,
    cross_support_ladder_round_trip,
    gravity_room_search,
    moves_used,
    run_actions,
)


def enter_level_9(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    solve.solve(env)


def result(env):
    return {
        "level": int(env.levels_completed) + 1,
        "terminal": bool(env.terminal()),
        "moves": moves_used(env.frame()),
    }


def probe(env):
    enter_level_9(env)
    for leg in (
        cross_persistent_support_rooms,
        cross_staged_gravity_zigzag,
        cross_support_ladder_round_trip,
    ):
        child = env.clone()
        returned = leg(child)
        print("REUSE", leg.__name__, returned, result(child))

    path = gravity_room_search(env, max_states=400, max_depth=48)
    child = env.clone()
    run_actions(child, path)
    print("SEARCH", len(path), path, result(child))


arena.run_program("bp35", probe)
