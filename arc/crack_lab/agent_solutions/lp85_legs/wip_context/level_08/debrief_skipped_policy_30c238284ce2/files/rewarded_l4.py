"""Reproduce the rewarded level-4 target pattern from observed permutations."""
import random
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players
from model_l4 import CLICKS, PAIR_INDICES, state


def reach(env):
    players.play_level_1(env)
    players.play_level_2(env)
    players.play_level_3(env)


def learn(env):
    rng = random.Random(854)
    samples = [[] for _ in CLICKS]
    for _ in range(300):
        node = env.clone()
        for _ in range(rng.randrange(9)):
            node.step(*CLICKS[rng.randrange(4)])
        before = state(node.frame())
        for op, action in enumerate(CLICKS):
            child = node.clone()
            child.step(*action)
            if child.levels_completed == env.levels_completed:
                samples[op].append((before, state(child.frame())))
    out = []
    for transitions in samples:
        perm = []
        for dest in range(36):
            candidates = [
                source for source in range(36)
                if all(after[dest] == before[source]
                       for before, after in transitions)
            ]
            if len(candidates) != 1:
                raise ValueError((dest, candidates))
            perm.append(candidates[0])
        out.append(tuple(perm))
    return tuple(out)


def apply(s, perm):
    return tuple(s[q] for q in perm)


def run(env):
    reach(env)
    start = state(env.frame())
    perms = learn(env)
    path = (1,) * 4 + (2,) * 8
    final = start
    for op in path:
        final = apply(final, perms[op])
    print("start", tuple(start[i:i+9] for i in range(0, 36, 9)))
    print("reward-target", tuple(final[i:i+9] for i in range(0, 36, 9)))
    print("scores", [
        sum(c[a] == c[b] for a, b in PAIR_INDICES)
        for c in (final[i:i+9] for i in range(0, 36, 9))
    ])
    clone = env.clone()
    for op in path:
        clone.step(*CLICKS[op])
    print("reward", clone.levels_completed, clone.terminal())


if __name__ == "__main__":
    print("run-result", A.run_program("lp85", run))
