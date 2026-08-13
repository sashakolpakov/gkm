"""Find and validate removable repeated-frame cycles in one level path."""

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena


BOUNDARIES = (0, 8, 42, 87, 149, 238, 331, 476, 544)


def frame_key(node):
    return np.asarray(node.frame())[1:, :].tobytes()


def observe(env):
    level = int(os.environ.get("TARGET_LEVEL", "7"))
    with open("checkpoint.json") as stream:
        admitted = json.load(stream)["final_path"]
    start, end = BOUNDARIES[level - 1], BOUNDARIES[level]
    for action in admitted[:start]:
        env.step(action)
    root = env.clone()
    segment = admitted[start:end]

    cursor = root.clone()
    occurrences = defaultdict(list)
    occurrences[frame_key(cursor)].append(0)
    for index, action in enumerate(segment, 1):
        cursor.step(action)
        occurrences[frame_key(cursor)].append(index)

    intervals = []
    for indices in occurrences.values():
        for left_index, first in enumerate(indices):
            for last in indices[left_index + 1:]:
                intervals.append((last - first, first, last))
    intervals.sort(reverse=True)

    tested = 0
    solution = None
    for saving, first, last in intervals[:500]:
        tested += 1
        candidate = root.clone()
        path = segment[:first] + segment[last:]
        for action in path:
            if candidate.terminal():
                break
            candidate.step(action)
        if candidate.levels_completed >= level:
            solution = (saving, first, last, path)
            break

    print("CYCLES", {"level": level, "repeated_intervals": len(intervals),
                     "tested": tested,
                     "saving": None if solution is None else solution[0],
                     "first": None if solution is None else solution[1],
                     "last": None if solution is None else solution[2],
                     "path": None if solution is None else solution[3]})


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path), "error": str(error)})
