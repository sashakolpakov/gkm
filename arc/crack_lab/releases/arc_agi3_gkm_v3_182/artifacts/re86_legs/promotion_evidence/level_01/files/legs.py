"""Reusable skills for selectable crosses and matching ring targets."""
from collections import Counter
import numpy as np


def align_selected_cross(env, stride=3):
    """Translate the selected cross onto its matching ring targets.

    The black center selects a cross; its four immediate neighbors reveal
    its color. Matching targets have eight color-4 pixels around that color.
    Repeated target rows and columns identify the cross's desired center.
    Progress is the number of matching targets lying on its finite arms.
    """
    frame = np.asarray(env.frame())
    centers = np.argwhere(frame == 0)
    if len(centers) != 1:
        raise ValueError('Expected one selected cross center')
    row, col = map(int, centers[0])
    color = int(frame[row - 1, col])
    targets = []
    for r, c in np.argwhere(frame == color):
        if 0 < r < 62 and 0 < c < 63:
            tile = frame[r-1:r+2, c-1:c+2]
            if np.count_nonzero(tile == 4) == 8:
                targets.append((int(r), int(c)))
    if len(targets) < 3:
        raise ValueError('Insufficient intact matching targets')
    target_row = Counter(r for r, c in targets).most_common(1)[0][0]
    target_col = Counter(c for r, c in targets).most_common(1)[0][0]
    dr, dc = target_row - row, target_col - col
    if dr % stride or dc % stride:
        raise ValueError('Target center is outside the movement lattice')
    path = ([1 if dr < 0 else 2] * (abs(dr) // stride)
            + [3 if dc < 0 else 4] * (abs(dc) // stride))
    for action in path:
        if env.terminal():
            return
        env.step(action)


def select_next_cross(env):
    """Cycle selection without translating either cross."""
    env.step(5)
