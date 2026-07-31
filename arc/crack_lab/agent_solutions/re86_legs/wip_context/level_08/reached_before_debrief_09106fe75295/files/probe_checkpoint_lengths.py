import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A


def probe(env):
    with open("checkpoint.json") as handle:
        path = json.load(handle)["final_path"]
    previous = env.levels_completed
    starts = [0]
    for index, action in enumerate(path, 1):
        env.step(action)
        if env.levels_completed != previous:
            print(
                "level",
                env.levels_completed,
                "segment",
                index - starts[-1],
                "cumulative",
                index,
            )
            starts.append(index)
            previous = env.levels_completed


if __name__ == "__main__":
    A.run_program("re86", probe)
