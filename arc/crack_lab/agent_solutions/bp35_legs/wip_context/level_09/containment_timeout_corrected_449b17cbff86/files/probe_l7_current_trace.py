"""Compact direct-replay trace of the currently composed level-7 leg."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import (
    ROW_ANCHORS,
    avatar_column,
    band_shift,
    cross_staged_gravity_zigzag,
)


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def controls(frame):
    return tuple(row for row in ROW_ANCHORS if int(frame[row][3]) == 8)


class TraceEnv:
    def __init__(self, env):
        self.env = env
        self.step_index = 0
        self.world_row = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, *action):
        before = self.env.frame()
        self.env.step(*action)
        self.step_index += 1
        if self.env.terminal():
            print("TRACE", self.step_index, action, "dead", self.env.levels_completed)
            return
        after = self.env.frame()
        shift = band_shift(before, after)
        self.world_row -= shift
        print(
            "TRACE",
            self.step_index,
            action,
            "world",
            self.world_row,
            "avatar",
            avatar_column(after),
            "shift",
            shift,
            "controls",
            controls(after),
            "target",
            any(int(value) == 7 for row in after for value in row),
        )


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)
    print("START", env.levels_completed, env.actions, avatar_column(env.frame()))
    result = cross_staged_gravity_zigzag(TraceEnv(env))
    print("END", result, env.levels_completed, env.terminal())


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
