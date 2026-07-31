import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from probe_optimize_level7 import completes, delete_chunks, runs, step_many
from search_l7_segments import PREFIX, SEGMENTS, expand

REDUCED = [
    (4, 10), (1, 7), (4, 1), (1, 4), (3, 6), (2, 1),
    (4, 6), (2, 7), (5, 1),
    (1, 3), (4, 3), (1, 2), (4, 5), (1, 6), (3, 12),
    (1, 4), (2, 4), (4, 13), (5, 1),
    (1, 1), (3, 2), (1, 8), (3, 3), (1, 3), (4, 3),
    (2, 4), (1, 3), (4, 6), (3, 6), (1, 4), (2, 3),
]


def route_runs(route):
    result = []
    start = 0
    for index, action in enumerate(route):
        if index == 0 or action != route[index - 1]:
            if index:
                result.append((start, index))
            start = index
    result.append((start, len(route)))
    return result


def probe(env):
    with open("checkpoint.json") as handle:
        full = json.load(handle)["final_path"]
    step_many(env, full[:PREFIX])
    root = env.clone()
    route = expand(REDUCED)
    print("start", len(route), completes(root, route), flush=True)
    iteration = 0
    while True:
        spans = route_runs(route)
        accepted = False
        tests = 0
        skip = 300 if iteration == 0 else 0
        for first in range(len(spans)):
            for second in range(first, len(spans)):
                if spans[first][0] == spans[first][1] - 1 and first == second:
                    continue
                remove = sorted(
                    (spans[first][1] - 1, spans[second][1] - 1),
                    reverse=True,
                )
                candidate = list(route)
                for index in remove:
                    candidate.pop(index)
                tests += 1
                if tests <= skip:
                    continue
                if completes(root, candidate):
                    route = delete_chunks(root, candidate)
                    iteration += 1
                    print(
                        "accept",
                        iteration,
                        first,
                        second,
                        "length",
                        len(route),
                        "tests",
                        tests,
                        flush=True,
                    )
                    print("accepted_runs", runs(route), flush=True)
                    accepted = True
                    break
                if tests % 50 == 0:
                    print("progress", iteration, tests, flush=True)
            if accepted:
                break
        if not accepted:
            break
    print("result", len(route), "runs", runs(route), flush=True)


if __name__ == "__main__":
    A.run_program("re86", probe)
