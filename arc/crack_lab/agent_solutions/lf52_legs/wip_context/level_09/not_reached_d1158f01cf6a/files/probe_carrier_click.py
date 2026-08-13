"""Test whether visible bordered carriers support coordinate movement."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_state, _movable_bridge_board
from perception import frame_delta


ENTRIES = {
    7: (331, (36, 36), ((36, 30), (36, 42), (30, 36), (42, 36))),
    9: (544, (36, 42), ((36, 48), (36, 54), (36, 60))),
}


def play(env, action):
    if isinstance(action, list):
        env.step(*action)
    else:
        env.step(action)


def summary(env):
    return {
        "bridge": _bridge_carrier_state(env.frame())[1:],
        "movable": _movable_bridge_board(env.frame())[1:],
    }


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        campaign = json.load(checkpoint_file)["final_path"]
    for level, (entry_index, source, destinations) in ENTRIES.items():
        node = env.clone()
        for action in campaign[:entry_index]:
            play(node, action)
        before_summary = summary(node)
        selected = node.clone()
        before = selected.frame()
        selected.step(6, source[1] + 1, source[0] + 1)
        print("SELECT", {
            "level": level,
            "source": source,
            "delta": {
                key: value
                for key, value in frame_delta(before, selected.frame()).items()
                if key != "samples"
            },
            "state_changed": summary(selected) != before_summary,
            "after": summary(selected),
        })
        for destination in destinations:
            child = node.clone()
            child.step(6, source[1] + 1, source[0] + 1)
            child.step(6, destination[1] + 1, destination[0] + 1)
            print("MOVE", {
                "level": level,
                "move": (source, destination),
                "state_changed": summary(child) != before_summary,
                "after": summary(child),
            })


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {
    "levels": levels,
    "moves": len(path),
    "error": str(error) if error else None,
})
