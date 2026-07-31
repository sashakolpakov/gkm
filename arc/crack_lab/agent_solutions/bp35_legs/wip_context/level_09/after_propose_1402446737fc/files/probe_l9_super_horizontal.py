"""Traverse the final shafts horizontally at different climb heights."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from probe_l9_boosted_supported_landing import report
from probe_l9_control_row import controls
from probe_l9_route_deletions import enter_level_9
from probe_l9_super_final_exit import super_boosted
from probe_l9_ultra_final_align import walk_left


def arrested(root):
    child = super_boosted(root)
    child.step(*controls(child)[0])
    child.step(3)
    child.step(6, 21, 33)
    child.step(6, 21, 35)
    for _ in range(3):
        child.step(6, 21, 33)
    child.step(6, 27, 27)
    child.step(4)
    child.step(*controls(child)[-1])
    for _ in range(8):
        child.step(6, 27, 33)
    child.step(6, 21, 45)
    child.step(*controls(child)[0])
    return child


def start(root, depth, column, height):
    child = arrested(root)
    for _ in range(depth):
        child.step(6, 27, 33)
    walk_left(child, column)
    child.step(*controls(child)[-1])
    x = 3 + 6 * column
    for _ in range(height):
        child.step(6, x, 33)
    return child


def probe(env):
    enter_level_9(env)
    depth, column, height = map(int, sys.argv[1:4])
    child = start(env, depth, column, height)
    report((depth, column, height, "START"), child)
    avatar_column = column
    for target in range(column + 1, 9):
        x = 3 + 6 * target
        color = int(child.frame()[39][x])
        print("TARGET", target, "color", color, flush=True)
        if color in (12, 14, 15):
            child.step(6, x, 39)
            report((target, "CLEAR"), child)
            if child.terminal():
                return
        elif color in (3, 5):
            return
        child.step(4)
        avatar_column = target
        report((avatar_column, "RIGHT"), child)
        if child.terminal() or int(child.levels_completed) >= 9:
            return


if __name__ == "__main__":
    arena.run_program("bp35", probe)
