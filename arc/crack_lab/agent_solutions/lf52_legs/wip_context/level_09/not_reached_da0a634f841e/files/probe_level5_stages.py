"""Show chosen and alternative bridge moves along the level-5 solution."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_moves, _bridge_carrier_state


ENTRY_INDEX = 149
EXIT_INDEX = 238


def play(env, action):
    if isinstance(action, list):
        env.step(*action)
    else:
        env.step(action)


def split_segments(actions):
    segments = []
    index = 0
    while index < len(actions):
        clicks = []
        while index < len(actions) and isinstance(actions[index], list):
            clicks.append(actions[index])
            index += 1
        keys = []
        while index < len(actions) and isinstance(actions[index], int):
            keys.append(actions[index])
            index += 1
        segments.append((clicks, keys))
    return segments


def compact_state(env):
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
        "legal": _bridge_carrier_moves(env.frame()),
    }


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        campaign = json.load(checkpoint_file)["final_path"]
    for action in campaign[:ENTRY_INDEX]:
        play(env, action)
    node = env.clone()
    print("ENTRY", compact_state(node))
    for number, (clicks, keys) in enumerate(
        split_segments(campaign[ENTRY_INDEX:EXIT_INDEX]), 1
    ):
        before = compact_state(node)
        for action in clicks:
            play(node, action)
        after_moves = compact_state(node)
        for action in keys:
            play(node, action)
        print("STAGE", {
            "number": number,
            "moves": [
                (
                    (clicks[index][2] - 1, clicks[index][1] - 1),
                    (clicks[index + 1][2] - 1, clicks[index + 1][1] - 1),
                )
                for index in range(0, len(clicks), 2)
            ],
            "keys": keys,
            "before": before,
            "after_moves": after_moves,
            "after_keys": compact_state(node),
        })


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {
    "levels": levels,
    "moves": len(path),
    "error": str(error) if error else None,
})
