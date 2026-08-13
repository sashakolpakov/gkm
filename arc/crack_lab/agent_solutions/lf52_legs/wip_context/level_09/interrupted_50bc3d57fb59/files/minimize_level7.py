"""Bounded reward-preserving subsequence minimization of level 7."""

import json
import math
import sys

sys.path.insert(0, "/private/tmp/gkm-submitted-protocol-lf52-20260805/arc/crack_lab")

import gkm_arena as arena

from perception import safe_step


PREFIX_END = 331
LEVEL_END = 476
MAX_CLONE_ACTIONS = 30000


def probe(env):
    with open("checkpoint.json") as handle:
        checkpoint = json.load(handle)
    full_path = checkpoint["final_path"]
    for action in full_path[:PREFIX_END]:
        env.step(action)
    root = env.clone()
    base_level = int(root.levels_completed)
    candidate = tuple(full_path[PREFIX_END:LEVEL_END])
    trials = 0
    clone_actions = 0

    def succeeds(actions):
        nonlocal trials, clone_actions
        if clone_actions + len(actions) > MAX_CLONE_ACTIONS:
            return False
        trials += 1
        node = root.clone()
        for action in actions:
            safe_step(node, action)
            clone_actions += 1
            if node.levels_completed > base_level:
                return True
            if node.terminal():
                break
        return False

    assert succeeds(candidate)
    granularity = 2
    while len(candidate) >= 2 and clone_actions < MAX_CLONE_ACTIONS:
        chunk = math.ceil(len(candidate) / granularity)
        reduced = False
        for start in range(0, len(candidate), chunk):
            trial = candidate[:start] + candidate[start + chunk:]
            if trial and succeeds(trial):
                candidate = trial
                granularity = max(2, granularity - 1)
                reduced = True
                print("REDUCED", len(candidate), "trials", trials, "steps", clone_actions, flush=True)
                break
        if reduced:
            continue
        if granularity >= len(candidate):
            break
        granularity = min(len(candidate), granularity * 2)

    changed = True
    while changed and clone_actions < MAX_CLONE_ACTIONS:
        changed = False
        for index in range(len(candidate)):
            trial = candidate[:index] + candidate[index + 1:]
            if trial and succeeds(trial):
                candidate = trial
                changed = True
                print("SINGLE", len(candidate), "trials", trials, "steps", clone_actions, flush=True)
                break

    print("MINIMIZED_LENGTH", len(candidate), "trials", trials, "steps", clone_actions)
    print("MINIMIZED_ACTIONS", json.dumps(candidate, separators=(",", ":")))


levels, path, error = arena.run_program("lf52", probe)
print("PROBE_RESULT", levels, len(path), error)
