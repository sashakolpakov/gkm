"""Learn level-5 tile permutations and search the compact symbolic state."""
from collections import deque
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from probe_l5 import ACTIONS, SLOTS, tile_state
from solve import solve

PAIRS = ((0, 4), (1, 3), (6, 8), (10, 12), (14, 16), (18, 20))

def infer_source_permutation(samples):
    source = []
    for p in range(len(SLOTS)):
        candidates = [
            q for q in range(len(SLOTS))
            if all(after[p] == before[q] for before, after in samples)
        ]
        if len(candidates) != 1:
            raise ValueError((p, candidates))
        source.append(candidates[0])
    return tuple(source)


def apply(state, source):
    return tuple(state[q] for q in source)


def score(state):
    """Dense progress: matched opposite tiles in the five horizontal motifs."""
    return sum(state[a] == state[b] for a, b in PAIRS)


def equality_key(state):
    labels = {}
    return tuple(labels.setdefault(value, len(labels)) for value in state)


def run(env):
    solve(env)
    start = tile_state(env.frame())
    contexts = []
    for a in range(5):
        for b in range(21):
            for c in range(5):
                contexts.append((ACTIONS[0],) * a + (ACTIONS[2],) * b
                                + (ACTIONS[0],) * c)
    samples = [[] for _ in ACTIONS]
    for path in contexts:
        node = env.clone()
        for action in path:
            node.step(*action)
        before = tile_state(node.frame())
        for i, action in enumerate(ACTIONS):
            child = node.clone()
            child.step(*action)
            samples[i].append((before, tile_state(child.frame())))
    perms = tuple(infer_source_permutation(s) for s in samples)
    print("start-score", score(start), "perms")
    for action, perm in zip(ACTIONS, perms):
        print(action[1:], perm)

    q = deque([(start, ())])
    seen = {equality_key(start)}
    best = (score(start), ())
    found = None
    while q:
        state, path = q.popleft()
        if score(state) > best[0]:
            best = (score(state), path)
            print("best", best[0], len(path), path, flush=True)
        if score(state) == len(PAIRS):
            found = path
            break
        if len(path) >= 16:
            continue
        for i, perm in enumerate(perms):
            child = apply(state, perm)
            key = equality_key(child)
            if key not in seen:
                seen.add(key)
                q.append((child, path + (i,)))
    print("searched", len(seen), "found", found, "best", best)
    if found is not None:
        clone = env.clone()
        for i in found:
            clone.step(*ACTIONS[i])
        print("verified", clone.levels_completed, clone.terminal(),
              tile_state(clone.frame()), score(tile_state(clone.frame())))


if __name__ == "__main__":
    print("run-result", A.run_program("lp85", run))
