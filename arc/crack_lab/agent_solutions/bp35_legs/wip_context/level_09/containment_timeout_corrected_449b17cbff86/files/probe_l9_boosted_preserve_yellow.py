"""Preserve both yellow supports during the boosted second reversal."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_preserve_boost import boosted
from probe_l9_route_deletions import enter_level_9
from probe_l9_supported_search import supported


def c4(root):
    child = boosted(root)
    child.step(*controls(child)[0])
    child.step(3)
    child.step(*controls(child)[0])
    child.step(4)
    return child


def probe(env):
    enter_level_9(env)
    for name, builder in (("CANONICAL_FIRST", supported), ("BOOSTED_FIRST", boosted)):
        first = builder(env)
        first.step(*controls(first)[0])
        report(name, first)
        print(
            name,
            "C3_COLORS",
            tuple((y, int(first.frame()[y][21])) for y in (21, 27, 33, 39, 45)),
            flush=True,
        )
        print(
            name,
            "LOW_CATCHES",
            tuple(
                (blob.bbox, blob.area)
                for blob in connected_components(
                    first.frame(), colors=(15,), min_area=2
                )
                if blob.bbox[0] >= 30 and blob.bbox[0] < 63
            ),
            flush=True,
        )
    child = c4(env)
    report("C4", child)
    for depth in range(1, 8):
        child.step(6, 27, 33)
        report(("DROP", depth), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            break


if __name__ == "__main__":
    arena.run_program("bp35", probe)
