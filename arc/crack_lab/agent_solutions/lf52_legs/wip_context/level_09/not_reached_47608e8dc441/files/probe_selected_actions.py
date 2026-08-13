"""Test directional and use actions while a legal level-9 piece is selected."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_state
from perception import frame_delta


def play(env, action):
    if isinstance(action, list):
        env.step(*action)
    else:
        env.step(action)


def compact(env):
    state = _bridge_carrier_state(env.frame())
    return {
        "pegs": sorted(state[1]),
        "carriers": sorted(state[2]),
        "bridges": sorted(state[3]),
        "selected": state[5],
        "level": env.levels_completed,
    }


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    with open("level9_candidate_102.json") as candidate_file:
        candidate = json.load(candidate_file)
    for action in prefix:
        play(env, action)
    entry = env.clone()
    source = [6, 19, 43]
    for action in (1, 2, 3, 4, 7):
        child = entry.clone()
        play(child, source)
        before = child.frame()
        child.step(action)
        print("ENTRY_SELECTED", {
            "action": action,
            "delta": {
                key: value
                for key, value in frame_delta(before, child.frame()).items()
                if key != "samples"
            },
            "state": compact(child),
        })
    remote = entry.clone()
    for action in candidate[:37]:
        play(remote, action)
    source = [6, 59, 19]
    for action in (1, 2, 3, 4, 7):
        child = remote.clone()
        play(child, source)
        before = child.frame()
        child.step(action)
        print("REMOTE_SELECTED", {
            "action": action,
            "delta": {
                key: value
                for key, value in frame_delta(before, child.frame()).items()
                if key != "samples"
            },
            "state": compact(child),
        })


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {
    "levels": levels,
    "moves": len(path),
    "error": str(error) if error else None,
})
