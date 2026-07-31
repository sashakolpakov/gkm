"""Stage a c8 lower catch by growing the neighboring c9 support downward."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_handoff_clickmap import avatar, handoff, relevant_full
from probe_l9_route_deletions import enter_level_9


def probe(env):
    enter_level_9(env)
    child = handoff(env)
    for label, action in (
        ("C9_Y29", (6, 57, 29)),
        ("C9_Y35", (6, 57, 35)),
        ("C9_Y41", (6, 57, 41)),
        ("C8_Y41", (6, 51, 41)),
        ("OPEN_C8", (6, 51, 35)),
    ):
        child.step(*action)
        report(label, child)
        print(
            "STATE",
            label,
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
        if child.terminal():
            return


if __name__ == "__main__":
    arena.run_program("bp35", probe)
