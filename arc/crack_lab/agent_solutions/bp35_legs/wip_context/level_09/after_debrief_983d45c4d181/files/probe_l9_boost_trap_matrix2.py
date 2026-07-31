"""Screen boosted switch/alignment/trapdoor choices with compact observations."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import controls
from probe_l9_oneflip_trapdoors import yellows
from probe_l9_preserve_boost import boosted
from probe_l9_route_deletions import enter_level_9


def boxes(env, color):
    return tuple(
        blob.bbox
        for blob in connected_components(env.frame(), colors=(color,), min_area=2)
        if blob.bbox[0] < 63
    )


def probe(env):
    enter_level_9(env)
    root = boosted(env)
    initial = controls(root)
    print("ROOT", "controls", initial, flush=True)
    for switch_index, switch in enumerate(initial):
        flipped = root.clone()
        flipped.step(*switch)
        for lefts in range(4):
            aligned = flipped.clone()
            for _ in range(lefts):
                aligned.step(3)
            before = controls(aligned)
            for yellow_index, yellow in enumerate(yellows(aligned)):
                child = aligned.clone()
                child.step(*yellow)
                print(
                    (switch_index, lefts, yellow_index),
                    "before",
                    before,
                    "yellow",
                    yellow,
                    "terminal",
                    bool(child.terminal()),
                    "levels",
                    int(child.levels_completed),
                    "after",
                    controls(child),
                    "avatar",
                    boxes(child, 9),
                    "goals",
                    boxes(child, 7),
                    flush=True,
                )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
