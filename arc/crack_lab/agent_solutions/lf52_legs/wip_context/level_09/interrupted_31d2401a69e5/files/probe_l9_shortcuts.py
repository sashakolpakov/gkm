"""Test whether removing the empty carrier bypasses level-9's relay phase."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import arr, connected_components, safe_step


ONE_PEG_MOVES = (
    ((42, 18), (42, 30)),
    ((48, 24), (36, 24)),
    ((42, 24), (30, 24)),
    ((36, 24), (24, 24)),
    ((30, 24), (18, 24)),
    ((18, 24), (18, 36)),
    ((18, 36), (30, 36)),
    ((24, 36), (36, 36)),
    ((30, 36), (42, 36)),
    ((36, 36), (48, 36)),
    ((48, 42), (48, 30)),
    ((48, 30), (36, 30)),
)


def observe(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)

    for rights in range(5):
        node = env.clone()
        for _ in range(rights):
            safe_step(node, 4)
        for source, destination in ONE_PEG_MOVES:
            safe_step(node, (6, source[1] + 1, source[0] + 1))
            safe_step(node, (6, destination[1] + 1, destination[0] + 1))
        pegs = tuple(
            blob.top_left for blob in connected_components(
                node.frame(), colors=(14,)
            ) if blob.area >= 8
        )
        print("BRANCH", {"rights": rights, "actions": rights + 24,
                         "level": node.levels_completed,
                         "terminal": node.terminal(), "pegs": pegs,
                         "carrier_pixels": int((arr(node.frame()) == 12).sum())})


levels, path, error = arena.run_program("lf52", observe)
print("PROBE_RESULT", {"levels": levels, "moves": len(path), "error": str(error)})
