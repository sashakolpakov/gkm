"""Tiny level-6 probes for socket-chain placements."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)


def replay(env, path):
    node = env.clone()
    for action in path:
        node.step(*action) if isinstance(action, tuple) else node.step(action)
    return node


def probe(env):
    for player in (
        play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
    ):
        player(env)
    candidates = {
        "B_down3_D_up2": (
            [(6, 45, 18)] + [2] * 3
            + [(6, 31, 46)] + [1] * 2 + [5]
        ),
        "B_down3_D_up3": (
            [(6, 45, 18)] + [2] * 3
            + [(6, 31, 46)] + [1] * 3 + [5]
        ),
        "B_down4_D_up2": (
            [(6, 45, 18)] + [2] * 4
            + [(6, 31, 46)] + [1] * 2 + [5]
        ),
        "A_down1_B_down3_D_up2": (
            [(6, 30, 19), 2]
            + [(6, 45, 18)] + [2] * 3
            + [(6, 31, 46)] + [1] * 2 + [5]
        ),
        "fill_A26_C35": (
            [(6, 30, 19)] + [2] * 3
            + [(6, 25, 33), 2, 5]
        ),
        "fill_B23_C35": (
            [(6, 45, 18)] + [2] * 3
            + [(6, 25, 33), 2, 5]
        ),
        "fill_A26_D35": (
            [(6, 30, 19)] + [2] * 3
            + [(6, 31, 46)] + [1] * 3 + [5]
        ),
        "fill_A26_C35_D35": (
            [(6, 30, 19)] + [2] * 3
            + [(6, 25, 33), 2]
            + [(6, 31, 46)] + [1] * 3 + [5]
        ),
    }
    print("CHAIN", {
        name: replay(env, path).levels_completed
        for name, path in candidates.items()
    })


if __name__ == "__main__":
    levels, path, err = A.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)
