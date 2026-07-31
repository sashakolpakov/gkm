"""Audit later route actions on top of the two verified early deletions."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_route_deletions import enter_level_9, replay, route, summary


def probe(env):
    enter_level_9(env)
    actions = route()
    target = summary(replay(env, actions))
    base_skips = {42, 48}
    combined = summary(replay(env, actions, skips=base_skips))
    print("COMBINED_EXACT", combined == target, combined)
    candidates = [
        index
        for index in range(49, len(actions))
        if index not in base_skips
    ]
    for ordinal, index in enumerate(candidates, 1):
        result = summary(replay(env, actions, skips=base_skips | {index}))
        if result == target:
            print("CANDIDATE", index, actions[index], result)
        if ordinal % 10 == 0:
            print("PROGRESS", ordinal, "of", len(candidates))


if __name__ == "__main__":
    arena.run_program("bp35", probe)
