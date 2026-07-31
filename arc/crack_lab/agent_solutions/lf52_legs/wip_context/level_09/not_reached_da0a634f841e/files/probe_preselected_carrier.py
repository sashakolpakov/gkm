"""Test whether a carrier alignment auto-completes a preselected peg jump."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _movable_bridge_board
from perception import frame_delta


def play(env, action):
    if isinstance(action, list):
        env.step(*action)
    else:
        env.step(action)


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        campaign = json.load(checkpoint_file)["final_path"]
    for action in campaign[:331]:
        play(env, action)
    node = env.clone()
    for action in (3, 3, 1, 1, 3, 3):
        node.step(action)
    print("BEFORE", _movable_bridge_board(node.frame()))
    node.step(6, 7, 13)
    selected_frame = node.frame()
    print("SELECTED", _movable_bridge_board(selected_frame))
    node.step(3)
    print("ALIGNED", {
        "board": _movable_bridge_board(node.frame()),
        "delta": {
            key: value
            for key, value in frame_delta(selected_frame, node.frame()).items()
            if key != "samples"
        },
        "level": node.levels_completed,
    })
    before_destination = node.frame()
    node.step(6, 7, 25)
    print("DESTINATION", {
        "board": _movable_bridge_board(node.frame()),
        "delta": {
            key: value
            for key, value in frame_delta(before_destination, node.frame()).items()
            if key != "samples"
        },
        "level": node.levels_completed,
    })


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {
    "levels": levels,
    "moves": len(path),
    "error": str(error) if error else None,
})
