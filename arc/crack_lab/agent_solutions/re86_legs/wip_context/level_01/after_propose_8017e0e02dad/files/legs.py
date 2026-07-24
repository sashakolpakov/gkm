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
