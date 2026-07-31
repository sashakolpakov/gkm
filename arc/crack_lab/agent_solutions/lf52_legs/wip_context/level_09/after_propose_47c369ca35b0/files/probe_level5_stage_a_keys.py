"""Trace the small opening carrier system one key at a time."""

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


def state(frame):
    slots, pegs, carriers, bridges, borders, selected = (
        _bridge_carrier_state(frame)
    )
    return (
        tuple(sorted(slots)),
        tuple(sorted(pegs)),
        tuple(sorted(carriers)),
        tuple(sorted(bridges)),
        tuple(sorted(borders)),
        selected,
    )


def probe(env):
    with open("campaign_candidate_633.json") as campaign_file:
        campaign = json.load(campaign_file)
    with open("level5_ddmin_89.json") as candidate_file:
        candidate = json.load(candidate_file)[:34]
    for action in campaign[:137]:
        play(env, action)
    node = env.clone()
    print("ENTRY", state(node.frame()), flush=True)
    for index, action in enumerate(candidate, 1):
        play(node, action)
        if isinstance(action, int) or index in (2, 9, 17, 34):
            print("STEP", index, action, state(node.frame()), flush=True)


if __name__ == "__main__":
    levels, path, error = arena.run_program("lf52", probe)
    print("HARNESS", {
        "levels": levels,
        "moves": len(path),
        "error": str(error) if error else None,
    })
