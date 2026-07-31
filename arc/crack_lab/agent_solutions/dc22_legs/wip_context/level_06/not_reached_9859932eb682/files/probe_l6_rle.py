"""Compact run-length map of the level-6 world, excluding the control panel."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve


def row_runs(values):
    runs = []
    start = 0
    value = int(values[0])
    for index, item in enumerate(values[1:], start=1):
        item = int(item)
        if item != value:
            runs.append((start, index - 1, value))
            start, value = index, item
    runs.append((start, len(values) - 1, value))
    return tuple(run for run in runs if run[2] != 4)


def observe(env):
    solve.solve(env)
    frame = perception.arr(env.frame())
    previous = None
    start = 0
    for row in range(63):
        runs = row_runs(frame[row, :40])
        if runs != previous:
            if previous:
                print("ROWS", (start, row - 1), previous)
            start, previous = row, runs
    if previous:
        print("ROWS", (start, 62), previous)


arena.run_program("dc22", observe)
