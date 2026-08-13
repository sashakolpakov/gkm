"""Replay the crossed level-7 relay into its final board."""

import json

import gkm_try

from perception import safe_step
from probe_level7_crossed_middle import CROSS_PREFIX
from probe_level7_crossed_keys import compact, legal_moves
from probe_level7_crossed_follow import (
    BRIDGE_LOAD,
    BRIDGE_LOAD_ALIGN,
    PEG_TRANSPORT,
    PEG_UNLOAD,
)


LEVEL_START = 331
BRIDGE_TRANSPORT = (1, 3, 3, 1, 1, 4, 4, 1, 1, 4, 4, 4, 2)
BRIDGE_UNLOAD = ((6, 47, 19), (6, 47, 31))
CROSSED_EXIT = (
    (6, 47, 25), (6, 47, 37),
    (6, 47, 31), (6, 47, 43),
    (6, 47, 37), (6, 59, 37),
)


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    for action in prefix[:LEVEL_START]:
        safe_step(env, action)
    actions = (
        CROSS_PREFIX + PEG_TRANSPORT + PEG_UNLOAD
        + BRIDGE_LOAD_ALIGN + BRIDGE_LOAD
        + BRIDGE_TRANSPORT + BRIDGE_UNLOAD + CROSSED_EXIT
    )
    snapshots = []
    for action in actions:
        before = compact(env)
        safe_step(env, action)
        after = compact(env)
        if before != after:
            snapshots.append((action, after))
    print("L7_CROSS_FINALBOARD", len(actions), int(env.levels_completed), compact(env), legal_moves(env))
    print("L7_CROSS_FINALBOARD_TAIL", snapshots[-8:])


if __name__ == "__main__":
    gkm_try.A.run_program("lf52", probe)
