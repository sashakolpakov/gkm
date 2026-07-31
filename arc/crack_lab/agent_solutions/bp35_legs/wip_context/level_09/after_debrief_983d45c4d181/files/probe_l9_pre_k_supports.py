"""Stage remote color-12 supports before the column-5 transition."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import _cell_shape, band_grid, click_action, moves_used
from perception import color_counts, connected_components
from probe_l9_top_handoff import enter_top


def enter_pre_k(env):
    enter_top(env)
    env.step(*click_action(3, 2))
    for col in (6, 5):
        env.step(*click_action(6, col))
        env.step(3)


def compact(env):
    blobs = connected_components(env.frame(), colors=(7, 8, 9, 12, 14, 15), min_area=3)
    return {
        "level": int(env.levels_completed) + 1,
        "terminal": bool(env.terminal()),
        "moves": moves_used(env.frame()),
        "colors": color_counts(env.frame()),
        "grid": tuple("".join(row) for row in band_grid(env.frame())),
        "shape5": tuple(_cell_shape(env.frame(), 5, col) for col in range(8)),
        "pieces": tuple(
            (b.color, b.bbox, b.area)
            for b in blobs
            if b.bbox[0] < 63 and b.color in (7, 8, 9, 12, 14)
        ),
    }


def probe(env):
    enter_pre_k(env)
    print("PRE_K", compact(env))
    for remote in range(8):
        if remote == 5:
            continue
        child = env.clone()
        child.step(*click_action(5, remote))
        print("REMOTE", remote, compact(child))
        child.step(*click_action(5, 5))
        print("ENTER", remote, compact(child))
    child = env.clone()
    for remote in (0, 1, 2, 3, 4, 6, 7):
        child.step(*click_action(5, remote))
    print("ALL_REMOTE", compact(child))
    child.step(*click_action(5, 5))
    print("ALL_ENTER", compact(child))


arena.run_program("bp35", probe)
