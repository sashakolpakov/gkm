"""Trace movable bridges, pegs, and carriers across an admitted level path."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _movable_bridge_board


BOUNDARIES = (0, 8, 42, 87, 149, 238, 331, 476, 544)


def observe(env):
    level = int(os.environ.get("TARGET_LEVEL", "7"))
    with open("checkpoint.json") as stream:
        path = json.load(stream)["final_path"]
    start, end = BOUNDARIES[level - 1], BOUNDARIES[level]
    for action in path[:start]:
        env.step(action)

    root_slots, root_carriers, root_bridges, root_pegs = \
        _movable_bridge_board(env.frame())
    print("ROOT", {"slots": tuple(sorted(root_slots)),
                   "carriers": tuple(sorted(root_carriers)),
                   "bridges": tuple(sorted(root_bridges)),
                   "pegs": tuple(sorted(root_pegs))})

    segment = path[start:end]
    index = 0
    macro_index = 0
    while index < len(segment):
        keys = []
        while index < len(segment) and not isinstance(segment[index], list):
            keys.append(segment[index])
            env.step(segment[index])
            index += 1
        if index >= len(segment):
            break
        macro = segment[index:index + 2]
        index += 2
        for action in macro:
            env.step(action)
        macro_index += 1
        _, carriers, bridges, pegs = _movable_bridge_board(env.frame())
        print("MACRO", {"index": macro_index, "keys": tuple(keys),
                        "move": tuple(tuple(action) for action in macro),
                        "carriers": tuple(sorted(carriers)),
                        "bridges": tuple(sorted(bridges)),
                        "pegs": tuple(sorted(pegs)),
                        "level": env.levels_completed})


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path), "error": str(error)})
