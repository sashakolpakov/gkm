"""Apply the supported high-high handoff from the four-control frontier."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_preserve_boost import boosted
from probe_l9_route_deletions import enter_level_9
from probe_l9_supported_high_path import high_high as canonical_high_high


def boxes(env, color):
    return tuple(
        blob.bbox
        for blob in connected_components(env.frame(), colors=(color,), min_area=2)
        if blob.bbox[0] < 63
    )


def report(label, env):
    print(
        label,
        "terminal",
        bool(env.terminal()),
        "levels",
        int(env.levels_completed),
        "avatar",
        boxes(env, 9),
        "controls",
        controls(env),
        "goals",
        boxes(env, 7),
        "grid",
        compact(env)["grid9"],
        flush=True,
    )


def boosted_high_high(root):
    child = boosted(root)
    child.step(*controls(child)[0])
    child.step(6, 21, 39)
    child.step(3)
    child.step(*controls(child)[0])
    return child


def lower_landing(root):
    child = boosted_high_high(root)
    child.step(4)
    for _ in range(5):
        child.step(6, 27, 33)
    child.step(6, 33, 27)
    child.step(4)
    child.step(*controls(child)[-1])
    return child


def probe(env):
    enter_level_9(env)
    canonical = canonical_high_high(env)
    report("CANONICAL_C3", canonical)
    canonical.step(4)
    report("CANONICAL_C4", canonical)
    child = boosted(env)
    report("BOOSTED", child)
    child = boosted_high_high(env)
    report("HIGH_HIGH", child)
    child.step(4)
    report("C4", child)
    for depth in range(1, 8):
        child.step(6, 27, 33)
        report(("DROP", depth), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            return


if __name__ == "__main__":
    arena.run_program("bp35", probe)
