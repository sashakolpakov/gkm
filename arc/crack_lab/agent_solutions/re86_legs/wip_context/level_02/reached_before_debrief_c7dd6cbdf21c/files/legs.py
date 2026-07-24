# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.

from collections import Counter, defaultdict

from perception import DOWN, LEFT, RIGHT, UP, USE, arr, connected_components


def _most_common(values):
    return Counter(values).most_common(1)[0][0]


def _move_on_lattice(env, row_delta, col_delta, step):
    if row_delta % step or col_delta % step:
        raise ValueError("cross target is off the movement lattice")

    vertical = DOWN if row_delta > 0 else UP
    horizontal = RIGHT if col_delta > 0 else LEFT
    for _ in range(abs(row_delta) // step):
        env.step(vertical)
    for _ in range(abs(col_delta) // step):
        env.step(horizontal)


def align_colored_crosses_to_ring_axes(env, ring_color=4):
    """Align movable coloured crosses with the axes marked by matching rings."""
    frame = arr(env.frame())
    rings = [
        blob
        for blob in connected_components(frame, colors=(ring_color,), min_area=8)
        if blob.size == (3, 3) and blob.area == 8
    ]
    if not rings:
        raise ValueError("no ring targets found")

    targets = defaultdict(list)
    ring_centers = set()
    for ring in rings:
        r0, c0, _, _ = ring.bbox
        center = (r0 + 1, c0 + 1)
        ring_centers.add(center)
        targets[int(frame[center])].append(center)

    crosses = {}
    for color, points in targets.items():
        pixels = [
            (int(r), int(c))
            for r, c in zip(*((frame == color).nonzero()))
            if (int(r), int(c)) not in ring_centers
        ]
        current = (
            _most_common(r for r, _ in pixels),
            _most_common(c for _, c in pixels),
        )
        target = (
            _most_common(r for r, _ in points),
            _most_common(c for _, c in points),
        )
        crosses[color] = (current, target)

    active = [
        color
        for color, (center, _) in crosses.items()
        if int(frame[center]) != color
    ]
    if len(active) != 1:
        raise ValueError("could not identify one active cross")

    step = rings[0].size[0]
    order = active + [color for color in crosses if color not in active]
    for index, color in enumerate(order):
        if index:
            env.step(USE)
        (row, col), (target_row, target_col) = crosses[color]
        _move_on_lattice(env, target_row - row, target_col - col, step)


def align_selected_outlines_to_ring_markers(env, ring_color=4):
    """Translate selectable coloured outlines through all matching ring centers."""
    frame = arr(env.frame())
    rings = [
        blob
        for blob in connected_components(frame, colors=(ring_color,), min_area=8)
        if blob.size == (3, 3) and blob.area == 8
    ]
    if not rings:
        raise ValueError("no ring markers found")

    ring_centers = {
        (ring.bbox[0] + 1, ring.bbox[1] + 1)
        for ring in rings
    }
    step = rings[0].size[0]
    background = _most_common(int(value) for value in frame.flat)
    scout = env.clone()
    placements = []
    seen_colors = set()

    while True:
        before = arr(scout.frame()).copy()
        black_pixels = list(zip(*((before == 0).nonzero())))
        if len(black_pixels) != 1:
            raise ValueError("could not identify selected outline center")
        current = tuple(int(value) for value in black_pixels[0])

        moved = scout.clone()
        moved.step(RIGHT)
        after = arr(moved.frame())
        erased = [
            (int(row), int(col), int(before[row, col]))
            for row, col in zip(*((before != after).nonzero()))
            if int(after[row, col]) == background
            and int(before[row, col]) not in (0, background, ring_color)
        ]
        if not erased:
            raise ValueError("selected outline did not move")
        color = _most_common(old for _, _, old in erased)
        if color in seen_colors:
            break
        seen_colors.add(color)

        offsets = {
            (row - current[0], col - current[1])
            for row, col, old in erased
            if old == color
        }
        radius = max(max(abs(row), abs(col)) for row, col in offsets)
        offsets.update(
            (int(row) - current[0], int(col) - current[1])
            for row, col in zip(*((before == color).nonzero()))
            if max(abs(int(row) - current[0]), abs(int(col) - current[1]))
            <= radius
        )
        targets = [
            point for point in ring_centers
            if int(frame[point]) == color
        ]
        candidates = [
            (row, col)
            for row in range(frame.shape[0])
            for col in range(frame.shape[1])
            if (row - current[0]) % step == 0
            and (col - current[1]) % step == 0
            and all((tr - row, tc - col) in offsets for tr, tc in targets)
        ]
        if not targets or not candidates:
            raise ValueError(f"no ring-fitting translation for color {color}")
        target = min(
            candidates,
            key=lambda point: (
                abs(point[0] - current[0]) + abs(point[1] - current[1]),
                point,
            ),
        )
        placements.append((current, target))
        scout.step(USE)

    for index, (current, target) in enumerate(placements):
        if index:
            env.step(USE)
        _move_on_lattice(
            env,
            target[0] - current[0],
            target[1] - current[1],
            step,
        )
