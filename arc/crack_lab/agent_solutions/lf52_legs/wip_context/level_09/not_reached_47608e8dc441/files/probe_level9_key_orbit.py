"""Map the valid horizontal key orbit after the level-9 entry relay."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from probe_level9_shortest_suffix import (
    dense_summary,
    physical_key,
    play,
)


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    with open("level9_candidate_102.json") as candidate_file:
        entry = json.load(candidate_file)[:28]
    for action in prefix + entry:
        play(env, action)
    node = env.clone()

    observations = []
    for step in range(1, 41):
        before = physical_key(node)
        play(node, 4)
        changed = physical_key(node) != before
        observations.append((step, changed, dense_summary(node)))
        if not changed:
            break
    print("RIGHT_ORBIT", observations, flush=True)


if __name__ == "__main__":
    levels, path, error = arena.run_program("lf52", probe)
    print("HARNESS", {
        "levels": levels,
        "moves": len(path),
        "error": str(error) if error else None,
    })
