"""Compact symbolic diagrams and tile signatures for freshly reached levels."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import players


CHAR = {0: "0", 2: "2", 4: ".", 5: "5", 6: "6",
        8: "8", 9: "9", 11: "B", 12: "C", 14: "E", 15: "F"}


def rows(frame, r0, r1, c0, c1):
    return [
        "".join(CHAR.get(int(value), "?") for value in row[c0:c1])
        for row in frame[r0:r1]
    ]


def probe(env):
    for level in range(1, 6):
        if level >= 2:
            frame = env.frame()
            print("LEVEL", level)
            print("top", *rows(frame, 0, 8, 0, 64), sep="\n")
            print("diagram", *rows(frame, 15, 43, 13, 51), sep="\n")
            print("palette", *rows(frame, 54, 63, 0, 64), sep="\n")
        if level == 5:
            return
        getattr(players, f"play_level_{level}")(env)


levels, path, err = A.run_program("sb26", probe)
print("done", levels, len(path), err)
