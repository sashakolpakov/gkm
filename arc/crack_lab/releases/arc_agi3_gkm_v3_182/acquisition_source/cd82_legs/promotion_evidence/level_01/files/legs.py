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
