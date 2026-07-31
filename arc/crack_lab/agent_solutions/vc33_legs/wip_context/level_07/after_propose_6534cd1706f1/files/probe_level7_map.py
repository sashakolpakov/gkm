"""Run-length map of the visible level-7 playfield."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import arr


with open("checkpoint.json") as checkpoint_file:
    PREFIX = json.load(checkpoint_file)["final_path"]


def probe(env):
    for action in PREFIX:
        env.step(*action)
    frame = arr(env.frame())
    rows = []
    for row in range(6, 58):
        runs = []
        start = 6
        color = int(frame[row, start])
        for column in range(7, 58):
            current = int(frame[row, column])
            if current != color:
                runs.append((start, column - 1, color))
                start, color = column, current
        runs.append((start, 57, color))
        rows.append(tuple(runs))

    start = 6
    previous = rows[0]
    for row, signature in enumerate(rows[1:], 7):
        if signature != previous:
            print((start, row - 1), previous)
            start, previous = row, signature
    print((start, 57), previous)


arena.run_program("vc33", probe)
