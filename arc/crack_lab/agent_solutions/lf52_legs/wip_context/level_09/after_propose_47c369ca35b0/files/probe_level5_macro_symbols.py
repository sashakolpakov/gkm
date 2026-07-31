"""Summarize the verified level-5 path as carrier and lattice macros."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from legs import _bridge_carrier_state


def play(env, action):
    if isinstance(action, list):
        env.step(*action)
    else:
        env.step(action)


def compact(frame):
    slots, pegs, carriers, bridges, borders, selected = (
        _bridge_carrier_state(frame)
    )
    return {
        "p": tuple(sorted(pegs)),
        "c": tuple(sorted(carriers)),
        "b": tuple(sorted(bridges)),
        "n": len(slots),
        "r": tuple(sorted(borders)),
        "s": selected,
    }


def probe(env):
    with open("campaign_candidate_633.json") as campaign_file:
        campaign = json.load(campaign_file)
    with open("level5_ddmin_89.json") as candidate_file:
        candidate = json.load(candidate_file)
    for action in campaign[:137]:
        play(env, action)
    node = env.clone()

    index = 0
    while index < len(candidate):
        if isinstance(candidate[index], int):
            start = index
            before = compact(node.frame())
            while index < len(candidate) and isinstance(candidate[index], int):
                play(node, candidate[index])
                index += 1
            print("ALIGN", {
                "range": (start + 1, index),
                "keys": tuple(candidate[start:index]),
                "before": before,
                "after": compact(node.frame()),
            }, flush=True)
            continue

        source_action = candidate[index]
        destination_action = candidate[index + 1]
        source = (source_action[2] - 1, source_action[1] - 1)
        destination = (
            destination_action[2] - 1,
            destination_action[1] - 1,
        )
        before = compact(node.frame())
        play(node, source_action)
        play(node, destination_action)
        print("MOVE", {
            "range": (index + 1, index + 2),
            "move": (source, destination),
            "before": before,
            "after": compact(node.frame()),
            "level": node.levels_completed,
        }, flush=True)
        index += 2


if __name__ == "__main__":
    levels, path, error = arena.run_program("lf52", probe)
    print("HARNESS", {
        "levels": levels,
        "moves": len(path),
        "error": str(error) if error else None,
    })
