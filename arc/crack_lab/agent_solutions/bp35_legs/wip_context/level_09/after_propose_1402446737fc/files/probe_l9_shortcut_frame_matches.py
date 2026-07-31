"""Find earlier known-route occurrences of the PRE_SECOND/right-end chamber."""

import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_presecond_outer import pre_second
from probe_l9_presecond_right_flip import right_end
from probe_l9_route_deletions import enter_level_9, route, step
from probe_l9_super_final_exit import SUPER_SKIPS


def normalized(env):
    frame = np.asarray(env.frame()).copy()
    frame[63, :] = 0
    frame[np.isin(frame, (8, 9, 11))] = 10
    return frame


def scan(root, name, actions, target):
    target_frame = normalized(target)
    child = root.clone()
    ranked = []
    for count, action in enumerate(actions, 1):
        step(child, action)
        diff = int(np.count_nonzero(normalized(child) != target_frame))
        ranked.append((diff, count, bool(child.terminal()), len(controls(child))))
        if child.terminal():
            break
    print(name, sorted(ranked)[:20], flush=True)


def probe(env):
    enter_level_9(env)
    early = env.clone()
    for _, action in route()[:21]:
        step(early, action)
    report("EARLY21", early)
    targets = {
        "PRE_SECOND": pre_second(env),
        "RIGHT_END": right_end(env),
    }
    original = [action for _, action in route()]
    compressed = [
        action
        for index, (_, action) in enumerate(route())
        if index not in SUPER_SKIPS
    ]
    for target_name, target in targets.items():
        report(("TARGET", target_name), target)
        print(
            "TARGET",
            target_name,
            "controls",
            len(controls(target)),
            flush=True,
        )
        scan(env, (target_name, "ORIGINAL"), original, target)
        scan(env, (target_name, "COMPRESSED"), compressed, target)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
