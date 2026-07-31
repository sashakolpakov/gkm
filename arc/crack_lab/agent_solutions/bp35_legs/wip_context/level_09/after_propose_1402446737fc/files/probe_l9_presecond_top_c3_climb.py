"""Use the yellow stopper aligned above c3 instead of the lethal open c2 shaft."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from legs import moves_used
from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_handoff_clickmap import avatar, relevant_full
from probe_l9_presecond_c9_top_cross import top_c9
from probe_l9_right_trap_stage import full_catches
from probe_l9_route_deletions import enter_level_9


def goals(env):
    return tuple(
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(7,), min_area=2)
        if blob.bbox[0] < 63
    )


def top_c3(root):
    child = top_c9(root)
    for x in (51, 45, 39, 33, 27, 21):
        child.step(6, x, 39)
        child.step(3)
    return child


def state(label, child):
    report(label, child)
    print(
        "STATE",
        label,
        "terminal",
        bool(child.terminal()),
        "moves",
        moves_used(child.frame()),
        "avatar",
        avatar(child),
        "controls",
        controls(child),
        "goals",
        goals(child),
        "full",
        full_catches(child),
        flush=True,
    )


def probe(env):
    enter_level_9(env)
    child = top_c3(env)
    state("TOP_C3", child)
    for label, action in (
        ("STAGE_C4", (6, 33, 33)),
        ("STAGE_C3", (6, 27, 33)),
        ("OPEN_CEILING", (6, 21, 33)),
        ("OPEN_STOPPER", (6, 21, 33)),
    ):
        child.step(*action)
        state(label, child)
        if child.terminal() or int(child.levels_completed) >= 9:
            return


if __name__ == "__main__":
    arena.run_program("bp35", probe)
