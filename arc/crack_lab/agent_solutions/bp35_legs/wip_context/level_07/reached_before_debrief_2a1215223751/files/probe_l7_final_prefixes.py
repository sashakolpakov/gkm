"""Compact rewind-prefix trace for the right-side corridor decode."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _cell_shape
from probe_l7_decode_matrix import build, controls, target
from probe_level7_reward_recovery import avatar_cell, lattice


FLAGS = (False, False, True, False, True)


def support_key(frame):
    return tuple(
        (i, j, _cell_shape(frame, i, j)[1])
        for i in range(10)
        for j in range(8)
        if _cell_shape(frame, i, j)[0] in (12, 14)
    )


def state(env):
    if env.terminal():
        return ("dead", int(env.levels_completed))
    frame = np.asarray(env.frame())
    return (
        "alive", int(env.levels_completed), avatar_cell(frame),
        target(frame), tuple(controls(frame)), support_key(frame),
        lattice(frame),
    )


def probe(env):
    with open("checkpoint.json") as stream:
        for action in json.load(stream)["final_path"]:
            env.step(action)
    with open("frontier_scaffold.json") as stream:
        raw = json.load(stream)["staged_prefix_actions"]
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
    previous = None
    for releases in range(int(os.environ.get("MAX_RELEASES", "21"))):
        current = state(env)
        if current != previous:
            shown = (
                current
                if os.environ.get("COMPACT") != "1"
                else current[:5] + (current[-1],)
            )
            print("FINAL_PREFIX", releases, shown, flush=True)
            previous = current
        if env.terminal():
            break
        env.step(7)


arena.run_program("bp35", probe)
