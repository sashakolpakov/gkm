"""Find when the first left ceiling handoff remains safe."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact
from probe_l9_early_flip_climb import enter_early_flip


def avatar(env):
    return [
        blob.bbox
        for blob in connected_components(env.frame(), colors=(9, 11), min_area=3)
        if blob.bbox[0] < 63
    ]


def probe(env):
    base = env.clone()
    for height in range(4):
        child = base.clone()
        enter_early_flip(child, height)
        before = compact(child)
        child.step(6, 51, 39)
        after_click = compact(child)
        click_terminal = bool(child.terminal())
        if not child.terminal():
            child.step(3)
        print(
            "HEIGHT",
            height,
            "before",
            before,
            "after_click",
            after_click,
            "click_terminal",
            click_terminal,
            "move_terminal",
            bool(child.terminal()),
            "avatar",
            avatar(child),
            "after_move",
            compact(child),
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
