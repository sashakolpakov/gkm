"""Open the supported right trapdoor and inspect its retained-control landing."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_oneflip_trapdoors import yellows
from probe_l9_preserve_boost import boosted
from probe_l9_route_deletions import enter_level_9


def full_catches(env):
    return tuple(
        (blob.color, blob.bbox, (6, round(blob.centroid[1]), round(blob.centroid[0])))
        for blob in connected_components(
            env.frame(), colors=(12, 14, 15), min_area=3
        )
        if blob.bbox[0] < 63 and blob.area == 21
    )


def supported(root):
    child = boosted(root)
    child.step(*controls(child)[0])
    child.step(3)
    child.step(3)
    child.step(*max(yellows(child), key=lambda action: action[1]))
    return child


def probe(env):
    enter_level_9(env)
    child = supported(env)
    report("SUPPORTED", child)
    print("FULL", full_catches(child), flush=True)

    descent = child.clone()
    descent.step(6, 21, 29)
    descent.step(4)
    report("RIGHT_LANDING", descent)
    for depth in range(1, 9):
        descent.step(6, 21, 33)
        report(("DESCENT", depth), descent)
        if descent.terminal() or int(descent.levels_completed) >= 9:
            break


if __name__ == "__main__":
    arena.run_program("bp35", probe)
