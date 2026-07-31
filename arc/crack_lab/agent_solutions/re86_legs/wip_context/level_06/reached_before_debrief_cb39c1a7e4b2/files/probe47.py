import json
import sys
from collections import Counter

sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A

from perception import arr

PATH = json.load(open('checkpoint.json'))['final_path']
UP, DOWN, LEFT, RIGHT, USE = 1, 2, 3, 4, 5
RECT = [UP] * 3 + [RIGHT] * 4 + [UP] * 8 + [RIGHT] * 9 + [DOWN] * 10
CROSS_STAGES = [
    ('top', [UP] * 2),
    ('left', [LEFT] * 7),
    ('down', [DOWN] * 3),
    ('right', [RIGHT] * 3),
    ('up', [UP] * 3),
    ('target-col', [LEFT] * 5),
    ('bottom-press', [DOWN] * 8),
    ('target-row', [UP] * 6),
]
MARK9 = {(6, 12), (9, 9), (9, 30), (27, 12)}


def geometry(env):
    frame = arr(env.frame())
    points = {
        (int(row), int(col))
        for row, col in zip(*(frame == 9).nonzero())
    } - MARK9
    points |= {
        (int(row), int(col))
        for row, col in zip(*(frame == 0).nonzero())
    }
    rows = Counter(row for row, _ in points)
    cols = Counter(col for _, col in points)
    center_row = rows.most_common(1)[0][0]
    center_col = cols.most_common(1)[0][0]
    vertical = [row for row, col in points if col == center_col]
    horizontal = [col for row, col in points if row == center_row]
    return (
        center_row,
        center_col,
        center_row - min(vertical),
        max(vertical) - center_row,
        center_col - min(horizontal),
        max(horizontal) - center_col,
    )


def solve(env):
    for action in PATH:
        env.step(action)
    clone = env.clone()
    for action in RECT:
        clone.step(action)
    clone.step(USE)
    for name, actions in CROSS_STAGES:
        for action in actions:
            clone.step(action)
        print(name, geometry(clone), 'level', clone.levels_completed)
    print('terminal', clone.terminal())


A.run_program('re86', solve)
