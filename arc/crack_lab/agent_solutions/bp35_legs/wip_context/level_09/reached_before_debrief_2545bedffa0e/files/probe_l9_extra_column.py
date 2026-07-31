"""Enter and climb the ninth, x=9 column revealed in the upper chamber."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import ROW_ANCHORS, cell_symbol, moves_used
from perception import connected_components
from probe_l9_k_room import enter_k_room


X_COLUMNS = (3, 9, 15, 21, 27, 33, 39, 45, 51, 57)


def compact(env):
    blobs = connected_components(env.frame(), colors=(7, 8, 9, 14, 15), min_area=3)
    return {
        "level": int(env.levels_completed) + 1,
        "terminal": bool(env.terminal()),
        "moves": moves_used(env.frame()),
        "grid9": tuple(
            "".join(cell_symbol(env.frame()[y][x]) for x in X_COLUMNS)
            for y in ROW_ANCHORS
        ),
        "pieces": tuple(
            (b.color, b.bbox, b.area)
            for b in blobs
            if b.bbox[0] < 63 and b.color in (7, 8, 9, 14)
        ),
    }


def probe(env):
    enter_k_room(env)
    env.step(6, 9, 33)
    env.step(6, 45, 33)
    print("READY", compact(env))
    for x in (39, 33, 27, 21, 15, 9, 3):
        env.step(6, x, 39)
        env.step(3)
        print("HANDOFF", x, compact(env))
        if env.terminal():
            return
    for advance in range(1, 5):
        env.step(6, 3, 33)
        print("CLIMB", advance, compact(env))
        if env.terminal() or int(env.levels_completed) >= 9:
            return


if __name__ == "__main__":
    arena.run_program("bp35", probe)
