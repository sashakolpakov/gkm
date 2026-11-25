"""Reusable skills inferred only from public frame/action experiments."""
from collections import deque

import numpy as np
from perception import arr, connected_components


def paint_half(env, side):
    """Aim the eight-position stencil and paint one half of a blank canvas.

    Arrow actions move the stencil toward their direction in 45-degree steps.
    Action 5 paints its facing half. Choose by observed clone paint coverage,
    since an arrow pointing opposite the current orientation can be inert.
    """
    frame = arr(env.frame())
    canvas = max(connected_components(frame, colors=[0]), key=lambda b: b.area)
    y0, x0, y1, x1 = canvas.bbox
    yy, xx = np.indices((y1 - y0 + 1, x1 - x0 + 1))
    desired = {
        'top': yy < (y1 - y0 + 1) / 2,
        'bottom': yy >= (y1 - y0 + 1) / 2,
        'left': xx < (x1 - x0 + 1) / 2,
        'right': xx >= (x1 - x0 + 1) / 2,
    }[side]
    queue = deque([(env.clone(), [])])
    seen = set()
    base_level = env.levels_completed
    while queue and len(seen) < 8:
        clone, path = queue.popleft()
        signature = (arr(clone.frame()) == 2).tobytes()
        if signature in seen:
            continue
        seen.add(signature)
        painted = clone.clone()
        painted.step(5)
        actual = arr(painted.frame())[y0:y1 + 1, x0:x1 + 1] == 15
        mismatch = int(np.count_nonzero(actual != desired))
        if painted.levels_completed > base_level or mismatch == 0:
            for action in path + [5]:
                env.step(action)
            return
        for action in (1, 2, 3, 4):
            turned = clone.clone()
            turned.step(action)
            queue.append((turned, path + [action]))
    raise RuntimeError('No observed stencil orientation paints the requested half')


def finish_reference_layer(env, color):
    """Select a swatch and finish a reference using one observed half-plane.

    Match the equally sized square reference and canvas, then test the eight
    stencil orientations on clones. Pixel mismatch supplies dense progress;
    commit only an exact match or a verified level completion.
    """
    frame = arr(env.frame())
    paint_colors = [0, 15, color]
    regions = connected_components(np.isin(frame, paint_colors).astype(int), colors=[1])
    squares = [b for b in regions if b.size[0] == b.size[1]
               and b.size[0] >= 6 and b.area == b.size[0] ** 2]
    pairs = [(a, b) for a in squares for b in squares
             if a.size == b.size and a.bbox[0] < b.bbox[0]]
    if len(pairs) != 1:
        raise RuntimeError('Expected one matching reference/canvas pair')
    reference, canvas = pairs[0]
    ry, rx, ry1, rx1 = reference.bbox
    y, x, y1, x1 = canvas.bbox
    target = frame[ry:ry1 + 1, rx:rx1 + 1].copy()
    swatches = [b for b in connected_components(frame, colors=[color])
                if b.size == (3, 3) and b.bbox[0] < y]
    if len(swatches) != 1:
        raise RuntimeError('Expected one palette swatch for the requested color')
    cy, cx = swatches[0].centroid
    select = (6, int(cx), int(cy))
    root = env.clone()
    root.step(*select)
    queue = deque([(root, [])])
    seen = set()
    base_level = env.levels_completed
    while queue and len(seen) < 8:
        clone, path = queue.popleft()
        signature = (arr(clone.frame()) == 2).tobytes()
        if signature in seen:
            continue
        seen.add(signature)
        painted = clone.clone()
        painted.step(5)
        mismatch = int(np.count_nonzero(
            arr(painted.frame())[y:y1 + 1, x:x1 + 1] != target))
        if painted.levels_completed > base_level or mismatch == 0:
            env.step(*select)
            for action in path + [5]:
                env.step(action)
            return
        for action in (1, 2, 3, 4):
            turned = clone.clone()
            turned.step(action)
            queue.append((turned, path + [action]))
    raise RuntimeError('No single observed half-plane completes the reference')
