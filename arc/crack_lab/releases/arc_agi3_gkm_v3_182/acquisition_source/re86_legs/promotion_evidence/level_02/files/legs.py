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


def align_selected_outline(env, stride=3):
    """Fit a translated visible outline over every matching ring center.

    Combine nearby clone views to recover pixels hidden by other outlines.
    The dense objective is matching targets covered by the translated body;
    verify full coverage on a clone before committing the translation.
    """
    frame = np.asarray(env.frame())
    row, col = map(int, np.argwhere(frame == 0)[0])
    moved = env.clone()
    moved.step(1)
    after = np.asarray(moved.frame())
    background = Counter(map(int, frame[:63].flat)).most_common(1)[0][0]
    changed = frame[:63][frame[:63] != after[:63]]
    color = Counter(int(v) for v in changed
                    if v not in (0, background)).most_common(1)[0][0]

    def targets(image):
        return {(int(r), int(c)) for r, c in np.argwhere(image == color)
                if 0 < r < image.shape[0] - 2 and 0 < c < image.shape[1] - 1
                and np.count_nonzero(image[r-1:r+2, c-1:c+2] == 4) == 8}

    goal = targets(frame)
    if not goal:
        return
    offsets = set()
    for action in (None, 1, 4):
        view = env.clone()
        if action is not None:
            view.step(action)
        image = np.asarray(view.frame())
        vr, vc = map(int, np.argwhere(image == 0)[0])
        isolated_targets = targets(image)
        offsets.update((int(r)-vr, int(c)-vc)
                       for r, c in np.argwhere(image == color)
                       if (int(r), int(c)) not in isolated_targets)
    first = min(goal)
    candidates = {(first[0]-dr, first[1]-dc) for dr, dc in offsets}
    candidates = [(r, c) for r, c in candidates
                  if 0 <= r < 63 and 0 <= c < 64
                  and (r-row) % stride == 0 and (c-col) % stride == 0
                  and all((tr-r, tc-c) in offsets for tr, tc in goal)]
    for r, c in sorted(candidates, key=lambda rc: abs(rc[0]-row)+abs(rc[1]-col)):
        dr, dc = r-row, c-col
        path = ([1 if dr < 0 else 2] * (abs(dr)//stride)
                + [3 if dc < 0 else 4] * (abs(dc)//stride))
        trial = env.clone()
        for action in path:
            trial.step(action)
        if trial.levels_completed > env.levels_completed or not targets(np.asarray(trial.frame())):
            for action in path:
                env.step(action)
            return
    raise ValueError('No verified translation covers the matching targets')
