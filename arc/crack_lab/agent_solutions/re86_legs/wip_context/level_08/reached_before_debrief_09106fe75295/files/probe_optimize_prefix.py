import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from probe_optimize_level7 import delete_chunks, erase_loops, runs


ENDS = (20, 56, 103, 171, 234, 306)


def probe(env):
    with open("checkpoint.json") as handle:
        full = json.load(handle)["final_path"]
    roots = [env.clone()]
    for index, action in enumerate(full[: ENDS[-1]], 1):
        env.step(action)
        if index in ENDS[:-1]:
            roots.append(env.clone())

    start = 0
    for level, (root, end) in enumerate(zip(roots, ENDS), 1):
        route = full[start:end]
        while True:
            route, removed = erase_loops(root, route)
            if removed is None:
                break
        route = delete_chunks(root, route)
        print(
            "optimized",
            level,
            "from",
            end - start,
            "to",
            len(route),
            "runs",
            runs(route),
            flush=True,
        )
        start = end


if __name__ == "__main__":
    A.run_program("re86", probe)
