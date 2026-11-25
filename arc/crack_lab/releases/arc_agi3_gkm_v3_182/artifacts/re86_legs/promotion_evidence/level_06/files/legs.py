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


def deform_selected_outline(env, approach, push_action, pushes, departure=()):
    """Push an outline against a solid obstacle, then leave by safe waypoints.

    Routes are (key, repetition) pairs. Obstacle contact can redistribute a
    rectangle's perimeter or slide a cross's strokes independently. Require
    a changed body relative to its selection marker on a clone before
    committing; ordinary translation alone does not satisfy this skill.
    Subsequent alignment can reuse the visible-outline matching skill.
    """
    from time import sleep

    def walk(trial, route):
        for action, count in route:
            for _ in range(count):
                sleep(0.004)
                trial.step(action)

    trial = env.clone()
    walk(trial, approach)
    before = np.asarray(trial.frame()).copy()
    walk(trial, ((push_action, pushes),))
    after = np.asarray(trial.frame())
    background = Counter(map(int, before[:63].flat)).most_common(1)[0][0]
    color = Counter(int(v) for v in before[:63][before[:63] != after[:63]]
                    if v not in (0, 4, background)).most_common(1)
    if not color:
        raise ValueError('Obstacle push did not change a visible outline')
    color = color[0][0]

    def body(image):
        cr, cc = map(int, np.argwhere(image == 0)[0])
        return {(int(r)-cr, int(c)-cc)
                for r, c in np.argwhere(image[:63] == color)
                if not (0 < r < 62 and 0 < c < 63
                        and np.count_nonzero(image[r-1:r+2,c-1:c+2] == 4) == 8)}

    if body(before) == body(after):
        raise ValueError('Obstacle push only translated the outline')
    walk(trial, departure)
    walk(env, tuple(approach) + ((push_action, pushes),) + tuple(departure))


def recolor_selected_outline(env, color, stride=3, return_to_start=False,
                             retreat_column=None):
    """Visit a bordered palette swatch, then retreat vertically into the board.

    Swatch interiors are uniform 4x4 blocks inside a color-2 border.
    Touching one with the moving outline changes its color. Verify moving
    body pixels on a clone after retreating, before committing the excursion.
    A retreat column can route a long outline around other paint swatches;
    optionally return to the original center to expose its full body again.
    """
    from time import sleep

    frame = np.asarray(env.frame())
    row, col = map(int, np.argwhere(frame == 0)[0])
    swatches = []
    for r in range(1, frame.shape[0]-5):
        for c in range(1, frame.shape[1]-5):
            tile = frame[r:r+6, c:c+6]
            if (np.all(tile[1:5, 1:5] == color)
                    and np.all(tile[[0, 5], :] == 2)
                    and np.all(tile[:, [0, 5]] == 2)):
                swatches.append((r+2.5, c+2.5))
    if not swatches:
        raise ValueError('No intact palette swatch for requested color')
    sr, sc = min(swatches, key=lambda p: abs(p[0]-row)+abs(p[1]-col))
    tr = row + stride*round((sr-row)/stride)
    tc = col + stride*round((sc-col)/stride)
    rc = tc if retreat_column is None else retreat_column
    path = ([1 if tr < row else 2] * (abs(tr-row)//stride)
            + [3 if tc < col else 4] * (abs(tc-col)//stride)
            + [3 if rc < tc else 4] * (abs(rc-tc)//stride)
            + [1 if row < tr else 2] * (abs(row-tr)//stride))
    if return_to_start:
        path += [3 if col < rc else 4] * (abs(col-rc)//stride)
    trial = env.clone()
    for action in path:
        sleep(0.004)
        trial.step(action)
    after = np.asarray(trial.frame())
    r, c = map(int, np.argwhere(after == 0)[0])
    shifted = trial.clone()
    sleep(0.004)
    shifted.step(1)
    shifted_frame = np.asarray(shifted.frame())
    if np.count_nonzero((after == color) & (shifted_frame != color)) < 4:
        raise ValueError('Palette visit did not recolor the selected outline')
    for action in path:
        env.step(action)


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


def align_outline_group(env, count, stride=3, selection_stride=1,
                        require_reward=True, horizontal_first=False):
    """Jointly translate same-color selectable outlines to cover their rings.

    Recover obscured body pixels from disappearing pixels in nearby clone
    views. Minimize total travel while maximizing the union of covered targets,
    then require a reward increase on a clone before committing the group.
    For a partial color group, require_reward=False instead checks every
    target's color and coverage. Selection stride skips other-color shapes;
    horizontal-first travel can avoid palette contact. Bodies may clip at
    the frame edges, while their selected centers must remain on the board.
    """
    from time import sleep

    frame_level = env.levels_completed
    directions = {1: (-stride, 0), 2: (stride, 0),
                  3: (0, -stride), 4: (0, stride)}

    def probe_step(trial, action):
        sleep(0.004)
        trial.step(action)

    def translation(start, end):
        dr, dc = end[0]-start[0], end[1]-start[1]
        vertical = [1 if dr < 0 else 2] * (abs(dr)//stride)
        horizontal = [3 if dc < 0 else 4] * (abs(dc)//stride)
        return horizontal + vertical if horizontal_first else vertical + horizontal

    frame = np.asarray(env.frame())
    height, width = frame.shape
    background = Counter(map(int, frame[:-1].flat)).most_common(1)[0][0]
    shifted = env.clone()
    probe_step(shifted, 1)
    after = np.asarray(shifted.frame())
    color = Counter(int(v) for v in frame[frame != after]
                    if v not in (0, 4, background)).most_common(1)[0][0]
    targets = [(int(r), int(c)) for r, c in np.argwhere(frame == color)
               if 0 < r < height-2 and 0 < c < width-1
               and np.count_nonzero(frame[r-1:r+2, c-1:c+2] == 4) == 8]
    if not targets:
        raise ValueError('No visible ring targets for the selected group')

    choices = []
    selection = env.clone()
    for index in range(count):
        center = tuple(map(int, np.argwhere(np.asarray(selection.frame()) == 0)[0]))
        offsets = {(0, 0)}
        for shift in (None, 1, 2, 3, 4):
            view = selection.clone()
            dr, dc = (0, 0) if shift is None else directions[shift]
            if shift is not None:
                probe_step(view, shift)
            vr, vc = center[0]+dr, center[1]+dc
            before = np.asarray(view.frame())
            for action in directions:
                moved = view.clone()
                probe_step(moved, action)
                after = np.asarray(moved.frame())
                erased = (before == color) & (after != color) & (after != 0)
                offsets.update((int(r)-vr, int(c)-vc)
                               for r, c in np.argwhere(erased))

        # Every useful placement maps some body pixel onto a ring center.
        positions = {(tr-dr, tc-dc) for tr, tc in targets for dr, dc in offsets}
        by_mask = {}
        for r, c in positions:
            if (r-center[0]) % stride or (c-center[1]) % stride:
                continue
            if not (0 <= r < height-1 and 0 <= c < width):
                continue
            mask = sum(1 << i for i, (tr, tc) in enumerate(targets)
                       if (tr-r, tc-c) in offsets)
            path = translation(center, (r, c))
            if mask not in by_mask or len(path) < len(by_mask[mask]):
                by_mask[mask] = path
        choices.append(by_mask)
        if index+1 < count:
            for _ in range(selection_stride):
                probe_step(selection, 5)

    plans = {0: []}
    for index, options in enumerate(choices):
        following = {}
        for covered, prefix in plans.items():
            for mask, moves in options.items():
                union = covered | mask
                path = prefix + ([5]*selection_stride if index else []) + moves
                if union not in following or len(path) < len(following[union]):
                    following[union] = path
        plans = following
    path = plans.get((1 << len(targets))-1)
    if path is None:
        raise ValueError('No joint translation covers the ring targets')
    trial = env.clone()
    for action in path:
        probe_step(trial, action)
        if trial.levels_completed > env.levels_completed:
            break
    if trial.levels_completed <= env.levels_completed:
        after = np.asarray(trial.frame())
        uncovered = any(after[r,c] not in (color, 0)
                        or np.count_nonzero(after[r-1:r+2,c-1:c+2] == 4) == 8
                        for r,c in targets)
        if require_reward or uncovered:
            raise ValueError('Joint coverage failed clone verification')
    for action in path:
        env.step(action)
        if env.levels_completed > frame_level:
            return
