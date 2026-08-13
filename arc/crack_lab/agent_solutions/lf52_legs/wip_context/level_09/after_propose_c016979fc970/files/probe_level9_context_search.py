"""Run the bounded level-9 search from a verified intermediate context."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from probe_level9_shortest_suffix import (
    dense_summary,
    play,
    search,
    visible_moves,
)


CONTEXT_INDEX = int(os.environ.get("CONTEXT_INDEX", "62"))
EXTRA_ACTIONS = json.loads(os.environ.get("EXTRA_ACTIONS", "[]"))


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    with open("level9_candidate_102.json") as candidate_file:
        candidate = json.load(candidate_file)
    for action in prefix:
        play(env, action)
    node = env.clone()
    for action in candidate[:CONTEXT_INDEX] + EXTRA_ACTIONS:
        play(node, action)
    print("CONTEXT", {
        "index": CONTEXT_INDEX,
        "extra": EXTRA_ACTIONS,
        "dense": dense_summary(node),
        "moves": visible_moves(node.frame()),
    }, flush=True)
    result = search(node)
    print("SEARCH", {
        "path": result[0],
        "cost": len(result[0]) if result[0] else None,
        "expanded": result[1],
        "seen": result[2],
        "best_dense": result[3],
    }, flush=True)


if __name__ == "__main__":
    levels, path, error = arena.run_program("lf52", probe)
    print("HARNESS", {
        "levels": levels,
        "moves": len(path),
        "error": str(error) if error else None,
    })
