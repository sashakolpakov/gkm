"""Check whether level-5 carrier controls are exactly reversible."""

import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena


CONTEXT_INDEX = 34


def play(env, action):
    if isinstance(action, list):
        env.step(*action)
    else:
        env.step(action)


def key(env):
    frame = np.asarray(env.frame()).copy()
    frame[0, :] = 0
    return frame.tobytes()


def probe(env):
    with open("campaign_candidate_633.json") as campaign_file:
        campaign = json.load(campaign_file)
    with open("level5_ddmin_89.json") as candidate_file:
        candidate = json.load(candidate_file)
    for action in campaign[:137] + candidate[:CONTEXT_INDEX]:
        play(env, action)

    results = []
    for action in (1, 2, 3, 4):
        inverse = {1: 2, 2: 1, 3: 4, 4: 3}[action]
        direct = env.clone()
        play(direct, action)
        if key(direct) == key(env):
            continue
        replayed = env.clone()
        play(replayed, action)
        play(replayed, inverse)
        restored = key(replayed) == key(env)
        play(replayed, action)
        same_child = key(replayed) == key(direct)
        future = []
        for next_action in (1, 2, 3, 4):
            left = replayed.clone()
            right = direct.clone()
            play(left, next_action)
            play(right, next_action)
            future.append((next_action, key(left) == key(right)))
        results.append((
            action, restored, same_child, tuple(future)
        ))
    print("UNDO_KEYS", results, flush=True)


if __name__ == "__main__":
    levels, path, error = arena.run_program("lf52", probe)
    print("HARNESS", {
        "levels": levels,
        "moves": len(path),
        "error": str(error) if error else None,
    })
