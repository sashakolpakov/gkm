"""Record compact world-pixel deltas along the validated prefix path."""

from collections import defaultdict
import json
import sys

import numpy as np

sys.path.insert(0, "/private/tmp/gkm-submitted-protocol-lf52-20260805/arc/crack_lab")

import gkm_arena as arena

from perception import arr, safe_step


def profile(env):
    with open("checkpoint.json") as handle:
        checkpoint = json.load(handle)
    noops = defaultdict(list)
    deltas = defaultdict(list)
    for absolute_index, action in enumerate(checkpoint["final_path"], 1):
        level = int(env.levels_completed) + 1
        before = arr(env.frame())[1:, :].copy()
        safe_step(env, action)
        count = int(np.count_nonzero(before != arr(env.frame())[1:, :]))
        deltas[level].append(count)
        if count == 0:
            noops[level].append((absolute_index, action))
    print("NOOPS", {level: values for level, values in sorted(noops.items())})
    print(
        "DELTA_COUNTS",
        {level: (len(values), values.count(0)) for level, values in sorted(deltas.items())},
    )


levels, path, error = arena.run_program("lf52", profile)
print("PROBE_RESULT", levels, len(path), error)
