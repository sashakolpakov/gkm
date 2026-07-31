"""Try verified suffix tails after each legal alternative checkpoint edge."""

import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from probe_level9_shortest_suffix import (
    move_actions,
    play,
    visible_moves,
)


CONTEXT_INDEX = int(os.environ.get("CONTEXT_INDEX", "62"))


def probe(env):
    with open("checkpoint.json") as checkpoint_file:
        prefix = json.load(checkpoint_file)["final_path"]
    with open("level9_candidate_102.json") as candidate_file:
        candidate = json.load(candidate_file)
    for action in prefix:
        play(env, action)
    root = env.clone()
    for action in candidate[:CONTEXT_INDEX]:
        play(root, action)

    alternatives = [
        (f"key:{action}", [action])
        for action in (3, 4)
    ]
    alternatives += [
        (
            f"{kind}:{source}->{destination}",
            list(move_actions(source, destination)),
        )
        for kind, source, destination in visible_moves(root.frame())
    ]
    boundaries = []
    index = CONTEXT_INDEX
    while index < len(candidate):
        boundaries.append(index)
        index += 1 if isinstance(candidate[index], int) else 2
    winners = []
    for label, edge in alternatives:
        branch = root.clone()
        for action in edge:
            play(branch, action)
        for start in boundaries:
            node = branch.clone()
            executed = []
            for action in candidate[start:]:
                play(node, action)
                executed.append(action)
                if node.levels_completed > 8:
                    break
            if node.levels_completed > 8:
                total = CONTEXT_INDEX + len(edge) + len(executed)
                winners.append((total, label, edge, start, executed))
                print("WIN", {
                    "total": total,
                    "label": label,
                    "edge": edge,
                    "suffix_start": start,
                    "suffix_cost": len(executed),
                }, flush=True)
    print("RESULT", {
        "alternatives": len(alternatives),
        "boundaries": len(boundaries),
        "winners": len(winners),
        "best": min(winners)[:4] if winners else None,
    }, flush=True)


if __name__ == "__main__":
    levels, path, error = arena.run_program("lf52", probe)
    print("HARNESS", {
        "levels": levels,
        "moves": len(path),
        "error": str(error) if error else None,
    })
