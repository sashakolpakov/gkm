"""Try verified level-5 suffix tails after each legal checkpoint edge."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_moves


CONTEXT_INDEX = int(os.environ.get("CONTEXT_INDEX", "34"))


def play(env, action):
    if isinstance(action, list):
        env.step(*action)
    else:
        env.step(action)


def physical_key(env):
    frame = np.asarray(env.frame()).copy()
    frame[0, :] = 0
    return frame.tobytes()


def move_actions(move):
    _, source, destination = move
    return (
        [6, source[1] + 1, source[0] + 1],
        [6, destination[1] + 1, destination[0] + 1],
    )


def probe(env):
    with open("campaign_candidate_633.json") as campaign_file:
        campaign = json.load(campaign_file)
    with open("level5_ddmin_89.json") as candidate_file:
        candidate = json.load(candidate_file)
    for action in campaign[:137]:
        play(env, action)
    root = env.clone()
    for action in candidate[:CONTEXT_INDEX]:
        play(root, action)

    alternatives = []
    root_key = physical_key(root)
    for action in (1, 2, 3, 4):
        child = root.clone()
        play(child, action)
        if physical_key(child) != root_key:
            alternatives.append((f"key:{action}", [action]))
    alternatives += [
        (
            f"{kind}:{source}->{destination}",
            list(move_actions((kind, source, destination))),
        )
        for kind, source, destination in _bridge_carrier_moves(root.frame())
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
                if node.levels_completed > 4:
                    break
            if node.levels_completed > 4:
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
        "context": CONTEXT_INDEX,
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
