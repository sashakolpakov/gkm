"""Test every key action at representative level-9 relay contexts."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_state
from perception import frame_delta


CONTEXTS = (0, 28, 37, 40, 62, 73, 84, 100)
KEYS = (1, 2, 3, 4, 7)


def play(env, action):
    if isinstance(action, list):
        env.step(*action)
    else:
        env.step(action)


def state_summary(env):
    slots, pegs, carriers, bridges, borders, selected = (
        _bridge_carrier_state(env.frame())
    )
    return {
        "slots": len(slots),
        "pegs": sorted(pegs),
        "carriers": sorted(carriers),
        "bridges": sorted(bridges),
        "borders": sorted(borders),
        "selected": selected,
    }


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    with open("level9_candidate_102.json") as candidate_file:
        candidate = json.load(candidate_file)
    for action in prefix:
        play(env, action)
    node = env.clone()
    current_index = 0
    for context in CONTEXTS:
        while current_index < context:
            play(node, candidate[current_index])
            current_index += 1
        before_summary = state_summary(node)
        results = {}
        for action in KEYS:
            child = node.clone()
            before = child.frame()
            child.step(action)
            after_summary = state_summary(child)
            results[action] = {
                "delta": {
                    key: value
                    for key, value in frame_delta(before, child.frame()).items()
                    if key != "samples"
                },
                "state_changed": after_summary != before_summary,
                "after": after_summary if after_summary != before_summary else None,
                "level": child.levels_completed,
            }
        print("CONTEXT", {
            "at": context,
            "state": before_summary,
            "actions": results,
        }, flush=True)


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {
    "levels": levels,
    "moves": len(path),
    "error": str(error) if error else None,
})
