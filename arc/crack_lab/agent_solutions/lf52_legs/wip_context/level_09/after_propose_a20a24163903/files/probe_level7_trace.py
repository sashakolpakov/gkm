"""Compact carrier/piece trace at every verified level-7 move landmark."""

import json

import gkm_try

from legs import _movable_bridge_board
from perception import safe_step
from probe_undo_slide import groups


LEVEL_START = 331
LEVEL_END = 476


def compact(node):
    slots, carriers, bridges, pegs = _movable_bridge_board(node.frame())
    return tuple(sorted(carriers)), tuple(sorted(bridges)), tuple(sorted(pegs))


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        path = json.load(checkpoint_file)["final_path"]
    for action in path[:LEVEL_START]:
        safe_step(env, action)
    print("L7_TRACE_ENTRY", compact(env))
    for index, (keys, pair) in enumerate(groups(path[LEVEL_START:LEVEL_END])):
        for action in keys:
            safe_step(env, action)
        before = compact(env)
        for action in pair:
            safe_step(env, action)
        print("L7_TRACE", index, keys, pair, before, compact(env), int(env.levels_completed))
        if int(env.levels_completed) >= 7:
            break


gkm_try.A.run_program("lf52", probe)
