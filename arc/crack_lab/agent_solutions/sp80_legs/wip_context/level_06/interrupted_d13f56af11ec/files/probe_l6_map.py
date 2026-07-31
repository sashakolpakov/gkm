"""Print a compact logical-cell map of the level-6 frame."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as Arena

import perception as P
from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)


CHARS = {1: "#", 4: "Y", 6: "M", 8: "o", 9: "A",
         11: "T", 12: ".", 14: "-", 15: "D"}


def probe(env):
    for player in (
        play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
    ):
        player(env)
    pixels = P.arr(env.frame())
    print("L6_LOGICAL_MAP")
    for top in range(2, 59, 3):
        cells = []
        for left in range(2, 59, 3):
            block = pixels[top:top + 3, left:left + 3]
            values, counts = __import__("numpy").unique(block, return_counts=True)
            color = int(values[counts.argmax()])
            cells.append(CHARS.get(color, "?"))
        print(f"{top:02d}", "".join(cells))


if __name__ == "__main__":
    levels, path, err = Arena.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)
