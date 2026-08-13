"""Follow the overlooked fixed-support jump near the remote relay edge."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_state
from probe_level9_shortest_suffix import (
    dense_summary,
    play,
    visible_moves,
)


def summary(node):
    state = _bridge_carrier_state(node.frame())
    return {
        "dense": dense_summary(node),
        "pegs": tuple(sorted(state[1])),
        "supports": tuple(sorted(state[3])),
        "moves": visible_moves(node.frame()),
        "level": node.levels_completed,
    }


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    with open("level9_candidate_102.json") as candidate_file:
        candidate = json.load(candidate_file)
    for action in prefix:
        play(env, action)
    node = env.clone()
    for action in candidate[:62]:
        play(node, action)
    direct = ([6, 17, 25], [6, 5, 25])
    for action in direct:
        play(node, action)
    print("DIRECT", summary(node), flush=True)
    for action in (1, 2, 3, 4):
        child = node.clone()
        play(child, action)
        print("DIRECT_KEY", action, summary(child), flush=True)
    hidden = node.clone()
    play(hidden, 4)
    for action in (1, 2, 3, 4):
        child = hidden.clone()
        play(child, action)
        print("HIDDEN_KEY", action, summary(child), flush=True)
    for count in range(1, 9):
        play(node, 4)
        print("RIGHT", count, summary(node), flush=True)


if __name__ == "__main__":
    levels, path, error = arena.run_program("lf52", probe)
    print("HARNESS", {
        "levels": levels,
        "moves": len(path),
        "error": str(error) if error else None,
    })
