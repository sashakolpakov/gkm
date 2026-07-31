"""Test direct and locally staged entries from c8 into the exterior c9 lane."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_handoff_clickmap import avatar, handoff, relevant_full
from probe_l9_route_deletions import enter_level_9


VARIANTS = {
    "DIRECT": (),
    "Y35": ((6, 57, 35),),
    "Y41": ((6, 57, 41),),
    "Y35_Y41": ((6, 57, 35), (6, 57, 41)),
    "Y41_Y35": ((6, 57, 41), (6, 57, 35)),
}


def probe(env):
    enter_level_9(env)
    root = handoff(env)
    for name, staging in VARIANTS.items():
        child = root.clone()
        for action in staging:
            child.step(*action)
        child.step(6, 57, 29)
        child.step(4)
        report(name, child)
        print(
            "STATE",
            name,
            "terminal",
            bool(child.terminal()),
            "avatar",
            avatar(child),
            "controls",
            controls(child),
            "full",
            relevant_full(child),
            flush=True,
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
