"""Clone-verify the best independently reward-checked campaign composition."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena


def play(env, action):
    if isinstance(action, list):
        env.step(*action)
    else:
        env.step(action)


def load(filename):
    with open(filename) as action_file:
        return json.load(action_file)


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        checkpoint = json.load(checkpoint_file)["final_path"]
    candidate = checkpoint[:87]
    candidate += load("level4_greedy_macro_candidate.json")
    candidate += checkpoint[149:238]
    candidate += load("level6_greedy_macro_candidate.json")
    candidate += load("level7_greedy_macro_candidate.json")
    candidate += load("level8_greedy_macro_candidate.json")
    candidate += load("level9_candidate_102.json")
    with open("campaign_candidate_632.json", "w") as candidate_file:
        json.dump(candidate, candidate_file, indent=2)
        candidate_file.write("\n")
    node = env.clone()
    transitions = []
    previous = node.levels_completed
    for index, action in enumerate(candidate, 1):
        play(node, action)
        if node.levels_completed != previous:
            transitions.append((node.levels_completed, index))
            previous = node.levels_completed
    print("VERIFY", {
        "actions": len(candidate),
        "levels": node.levels_completed,
        "transitions": transitions,
    })


levels, path, error = arena.run_program("lf52", probe)
print("HARNESS", {
    "levels": levels,
    "moves": len(path),
    "error": str(error) if error else None,
})
