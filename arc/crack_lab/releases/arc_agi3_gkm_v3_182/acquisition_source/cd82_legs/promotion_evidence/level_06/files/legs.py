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


def finish_stencil_reference(env):
    """Match a reference by peeling observed half-plane and small-stamp layers.

    Later layers may cover earlier ones. In reverse order, a usable stamp has
    one target color on every pixel not already covered by a later layer.
    Verify the entire resulting composition on a clone before committing it.
    """
    import time
    from perception import safe_step

    def trial(source, actions):
        clone = source.clone()
        for action in actions:
            safe_step(clone, action)
            time.sleep(.004)
        return clone

    frame = arr(env.frame())
    swatches = {b.color: (6, int(b.centroid[1]), int(b.centroid[0]))
                for b in connected_components(frame)
                if b.size == (3, 3) and b.area == 9
                and b.bbox[0] < 8 and b.bbox[1] >= 18}
    regions = connected_components(
        np.isin(frame, list(swatches) + [0]).astype(int), colors=[1])
    squares = [b for b in regions if b.size[0] == b.size[1]
               and b.size[0] >= 6 and b.area == b.size[0] ** 2]
    pairs = [(a, b) for a in squares for b in squares
             if a.size == b.size and a.bbox[0] < b.bbox[0]]
    if len(pairs) != 1:
        raise RuntimeError('Expected one reference/canvas pair')
    reference, canvas = pairs[0]
    ry, rx, ry1, rx1 = reference.bbox
    y, x, y1, x1 = canvas.bbox
    target = frame[ry:ry1 + 1, rx:rx1 + 1].copy()
    base = frame[y:y1 + 1, x:x1 + 1].copy()
    marker = next(c for c in swatches if c not in base)
    signature = lambda c: (arr(c.frame()) == 2).tobytes()
    queue = deque([(env.clone(), [])])
    seen, operations = set(), []
    while queue and len(seen) < 8:
        clone, path = queue.popleft()
        key = signature(clone)
        if key in seen:
            continue
        seen.add(key)
        blobs = connected_components(clone.frame(), colors=[2])
        small = min(blobs, key=lambda b: b.area)
        cy, cx = small.centroid
        for action in (5, (6, int(cx), int(cy))):
            painted = trial(clone, [swatches[marker], action])
            mask = arr(painted.frame())[y:y1 + 1, x:x1 + 1] == marker
            if mask.any():
                operations.append((mask, key, action))
        for action in (1, 2, 3, 4):
            queue.append((trial(clone, [action]), path + [action]))

    covered = np.zeros(target.shape, dtype=bool)
    layers = []
    while np.any((target != base) & ~covered):
        options = []
        for mask, key, action in operations:
            colors = np.unique(target[mask & ~covered])
            if len(colors) == 1 and int(colors[0]) in swatches:
                options.append((int(np.sum(mask & ~covered)),
                                int(colors[0]), mask, key, action))
        if not options:
            raise RuntimeError('Observed stencil masks cannot compose the reference')
        _, color, mask, key, action = max(options, key=lambda item: item[0])
        covered |= mask
        layers.append((color, key, action))

    clone, solution = env.clone(), []
    level = env.levels_completed
    for color, goal_key, stamp in reversed(layers):
        queue = deque([(clone, [])])
        seen = set()
        while queue:
            candidate, path = queue.popleft()
            key = signature(candidate)
            if key == goal_key:
                break
            if key in seen:
                continue
            seen.add(key)
            for action in (1, 2, 3, 4):
                queue.append((trial(candidate, [action]), path + [action]))
        else:
            raise RuntimeError('Stencil orientation became unreachable')
        actions = path + [swatches[color], stamp]
        clone = trial(clone, actions)
        solution.extend(actions)
        if clone.levels_completed > level:
            break
    if clone.levels_completed <= level:
        raise RuntimeError('Layer composition failed clone verification')
    for action in solution:
        safe_step(env, action)
