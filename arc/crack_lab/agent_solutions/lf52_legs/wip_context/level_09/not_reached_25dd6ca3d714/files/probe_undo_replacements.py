"""Find short undo-assisted replacements for long admitted key windows."""

import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena


BOUNDARIES = (0, 8, 42, 87, 149, 238, 331, 476, 544)
KEYS = (1, 2, 3, 4, 7)
REPLACEMENTS = ((7,),) + tuple(
    pair for pair in itertools.product(KEYS, repeat=2) if 7 in pair
)


def frame_key(node):
    return np.asarray(node.frame())[1:, :].tobytes()


def observe(env):
    level = int(os.environ.get("TARGET_LEVEL", "7"))
    with open("checkpoint.json") as stream:
        admitted = json.load(stream)["final_path"]
    start, end = BOUNDARIES[level - 1], BOUNDARIES[level]
    for action in admitted[:start]:
        env.step(action)

    segment = admitted[start:end]
    cursor = env.clone()
    trials = 0
    found = []
    for first, action in enumerate(segment):
        if isinstance(action, list):
            cursor.step(action)
            continue
        run_end = first
        while run_end < len(segment) and not isinstance(segment[run_end], list):
            run_end += 1
        max_window = min(10, run_end - first)
        for length in range(max_window, 2, -1):
            baseline = cursor.clone()
            for known_action in segment[first:first + length]:
                baseline.step(known_action)
            target = frame_key(baseline)
            for replacement in REPLACEMENTS:
                if len(replacement) >= length:
                    continue
                trials += 1
                candidate = cursor.clone()
                for replacement_action in replacement:
                    candidate.step(replacement_action)
                if frame_key(candidate) != target:
                    continue
                for suffix_action in segment[first + length:]:
                    if candidate.terminal():
                        break
                    candidate.step(suffix_action)
                if candidate.levels_completed >= level:
                    path = (segment[:first] + list(replacement)
                            + segment[first + length:])
                    found.append((length - len(replacement), first, length,
                                  replacement, path))
                    print("FOUND", {"saving": length - len(replacement),
                                    "first": first, "old": tuple(
                                        segment[first:first + length]
                                    ), "replacement": replacement,
                                    "new_actions": len(path)})
        cursor.step(action)

    if found:
        best = max(found, key=lambda item: item[0])
        print("BEST", {"saving": best[0], "first": best[1],
                       "length": best[2], "replacement": best[3],
                       "path": best[4], "trials": trials})
    else:
        print("BEST", {"saving": 0, "trials": trials})


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path), "error": str(error)})
