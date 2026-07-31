"""Map one-click staging options at the retained-control c8 handoff."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components, frame_delta
from probe_l9_boosted_supported_landing import boxes
from probe_l9_control_row import controls
from probe_l9_presecond_right_handoff import landing
from probe_l9_right_trap_stage import full_catches
from probe_l9_route_deletions import enter_level_9


def relevant_full(env):
    return tuple(
        action
        for _, _, action in full_catches(env)
        if action[1] >= 33 and action[2] >= 27
    )


def avatar(env):
    return boxes(env, 9) + boxes(env, 11)


def handoff(root):
    child = landing(root, 0)
    child.step(6, 51, 27)
    child.step(4)
    return child


def probe(env):
    enter_level_9(env)
    base = handoff(env)
    candidates = relevant_full(base)
    print(
        "BASE",
        "terminal",
        bool(base.terminal()),
        "avatar",
        avatar(base),
        "controls",
        controls(base),
        "full",
        candidates,
        flush=True,
    )
    before = base.frame()
    for action in candidates:
        staged = base.clone()
        staged.step(*action)
        delta = frame_delta(before, staged.frame())
        after_full = relevant_full(staged)
        descended = staged.clone()
        descended.step(6, 51, 35)
        print(
            "CLICK",
            action,
            "delta",
            (delta["count"], delta["bbox"]),
            "full",
            after_full,
            "then_open",
            (
                bool(descended.terminal()),
                avatar(descended),
                controls(descended),
                relevant_full(descended),
            ),
            flush=True,
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
