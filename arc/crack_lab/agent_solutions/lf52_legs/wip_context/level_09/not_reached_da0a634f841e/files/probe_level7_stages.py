"""Summarize level-7 carrier segments and puzzle pieces."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _movable_bridge_board


ENTRY_INDEX = 331
EXIT_INDEX = 476


def play(env, action):
    if isinstance(action, list):
        env.step(*action)
    else:
        env.step(action)


def split_segments(actions):
    segments = []
    index = 0
    while index < len(actions):
        keys = []
        while index < len(actions) and isinstance(actions[index], int):
            keys.append(actions[index])
            index += 1
        clicks = []
        while index < len(actions) and not isinstance(actions[index], int):
            clicks.append(actions[index])
            index += 1
        segments.append((keys, clicks))
    return segments


def summary(env):
    slots, carriers, bridges, pegs = _movable_bridge_board(env.frame())
    return {
        "level": env.levels_completed,
        "slots": len(slots),
        "carriers": sorted(carriers),
        "bridges": sorted(bridges),
        "pegs": sorted(pegs),
    }


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        campaign = json.load(checkpoint_file)["final_path"]
    for action in campaign[:ENTRY_INDEX]:
        play(env, action)
    node = env.clone()
    print("ENTRY", summary(node))
    for number, (keys, clicks) in enumerate(
        split_segments(campaign[ENTRY_INDEX:EXIT_INDEX]), 1
    ):
        for action in keys:
            play(node, action)
        after_keys = summary(node)
        for action in clicks:
            play(node, action)
        print("SEGMENT", {
            "number": number,
            "keys": keys,
            "moves": len(clicks) // 2,
            "after_keys": after_keys,
            "after_moves": summary(node),
        })


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {
    "levels": levels,
    "moves": len(path),
    "error": str(error) if error else None,
})
