"""Pre-stage the off-screen c9 lower catch before reversing gravity."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_handoff_clickmap import avatar
from probe_l9_presecond_right_flip import right_end
from probe_l9_right_trap_stage import full_catches
from probe_l9_route_deletions import enter_level_9


STAGING = (
    (6, 45, 45),
    (6, 45, 51),
    (6, 45, 57),
    (6, 51, 57),
    (6, 57, 57),
)


def staged(root):
    child = right_end(root)
    for action in STAGING:
        child.step(*action)
    return child


def state(label, child):
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
        full_catches(child),
        flush=True,
    )


def probe(env):
    enter_level_9(env)
    child = right_end(env)
    state("ROOT", child)
    for index, action in enumerate(STAGING, 1):
        child.step(*action)
        state(("STAGE", index, action), child)
        if child.terminal():
            return
    child.step(*controls(child)[0])
    child.step(6, 45, 33)
    child.step(6, 45, 33)
    child.step(6, 51, 27)
    child.step(4)
    child.step(6, 57, 29)
    child.step(4)
    state("C9", child)
    child.step(6, 57, 35)
    state("C9_DROP1", child)


if __name__ == "__main__":
    arena.run_program("bp35", probe)
