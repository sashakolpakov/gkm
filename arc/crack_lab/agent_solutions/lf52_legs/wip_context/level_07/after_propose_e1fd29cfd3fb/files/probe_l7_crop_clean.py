import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr
from probe_l7_relay import setup


CHARS = {
    0: ".",
    1: "o",
    5: "#",
    7: "g",
    8: "B",
    9: "+",
    10: " ",
    11: "c",
    12: "C",
    14: "P",
    15: "x",
}


def crop(frame, rows, columns):
    grid = arr(frame)
    return tuple(
        "".join(CHARS[int(grid[row, column])] for column in columns)
        for row in rows
    )


def probe(env):
    setup(env)
    for name, rows, columns in (
        ("TOP", range(3, 31), range(0, 64)),
        ("BOTTOM", range(29, 61), range(0, 64)),
    ):
        print(name)
        for row, line in zip(rows, crop(env.frame(), rows, columns)):
            print(f"{row:02d}", line)


levels, path, error = arena.run_program("lf52", probe)
print("RESULT", levels, len(path), error)
