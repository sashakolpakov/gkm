import json
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import bounded_bfs


def key(node):
    frame = np.asarray(node.frame())[:62]
    return np.where(np.isin(frame, (1, 8, 9, 11, 14, 15)), frame, 0).tobytes()


def probe(env):
    with open("checkpoint.json") as fh:
        path = json.load(fh)["final_path"]
    for action in path:
        env.step(action)
    started = time.time()
    plan = bounded_bfs(
        env,
        lambda node, path: int(node.levels_completed) > 6,
        key_fn=key,
        max_states=100000,
        max_depth=120,
    )
    print("search", round(time.time() - started, 3), plan)


levels, path, err = arena.run_program("g50t", probe)
print("probe_result", levels, len(path), err)
