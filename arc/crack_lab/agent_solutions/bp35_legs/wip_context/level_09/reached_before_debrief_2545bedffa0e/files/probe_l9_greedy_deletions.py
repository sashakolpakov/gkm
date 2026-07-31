"""Compose only exact-endpoint deletions from the promising compression cluster."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_route_deletions import enter_level_9, replay, route, summary


def probe(env):
    enter_level_9(env)
    actions = route()
    target = summary(replay(env, actions))
    skips = {42, 48}
    for index in (62, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74):
        candidate = skips | {index}
        result = summary(replay(env, actions, skips=candidate))
        accepted = result == target
        print(
            "TRY",
            index,
            actions[index],
            "accepted",
            accepted,
            "terminal",
            result["terminal"],
            "avatar",
            result["avatar"],
            "controls",
            len(result["controls"]),
        )
        if accepted:
            skips = candidate
    print("SKIPS", sorted(skips), "SAVED", len(skips))
    print("FINAL", summary(replay(env, actions, skips=skips)))


if __name__ == "__main__":
    arena.run_program("bp35", probe)
