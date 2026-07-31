"""Greedily remove every action that preserves the chamber-two endpoint."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_route_deletions import enter_level_9, replay, route, summary
from probe_l9_twelve_fast_frontier import SKIPS


def probe(env):
    enter_level_9(env)
    actions = route()
    target = summary(replay(env, actions))
    skips = set(SKIPS)
    for ordinal, index in enumerate(range(len(actions)), 1):
        if index in skips:
            continue
        result = summary(replay(env, actions, skips=skips | {index}))
        if result == target:
            skips.add(index)
            print("ACCEPT", index, actions[index], "saved", len(skips), flush=True)
        if ordinal % 10 == 0:
            print("PROGRESS", ordinal, "of", len(actions), flush=True)
    print("SKIPS", sorted(skips), flush=True)
    print("SAVED", len(skips), "REMAINING", len(actions) - len(skips), flush=True)
    print("FINAL", summary(replay(env, actions, skips=skips)), flush=True)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
