"""Clone-verify the preserved optimized campaign candidate from level 1."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena


def probe(env):
    with open("campaign_candidate_633.json") as candidate_file:
        candidate = json.load(candidate_file)
    clone = env.clone()
    transitions = []
    previous = clone.levels_completed
    for index, action in enumerate(candidate, 1):
        clone.step(action)
        if clone.levels_completed != previous:
            transitions.append((clone.levels_completed, index))
            previous = clone.levels_completed
    print("VERIFY", {
        "actions": len(candidate),
        "levels": clone.levels_completed,
        "transitions": transitions,
    })


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {"levels": levels, "moves": len(path), "error": str(error) if error else None})
