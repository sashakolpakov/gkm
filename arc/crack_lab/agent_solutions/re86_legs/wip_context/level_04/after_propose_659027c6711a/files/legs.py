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


def cover_ring_markers_with_selected_shapes(env, ring_color=4):
    """Translate selectable line/X/diamond shapes to cover matching ring centers."""
    frame = arr(env.frame())
    rings = [
        blob
        for blob in connected_components(frame, colors=(ring_color,), min_area=8)
        if blob.size == (3, 3) and blob.area == 8
    ]
    if not rings:
        raise ValueError("no ring markers found")

    targets = {
        (ring.bbox[0] + 1, ring.bbox[1] + 1)
        for ring in rings
    }
    target_colors = {int(frame[point]) for point in targets}
    if len(target_colors) != 1:
        raise ValueError("selected-shape leg requires one marker color")
    shape_color = target_colors.pop()
    background = _most_common(int(value) for value in frame.flat)
    step = rings[0].size[0]

    shapes = []
    scout = env.clone()
    first_center = None
    while True:
        before = arr(scout.frame()).copy()
        black_pixels = list(zip(*((before == 0).nonzero())))
        if len(black_pixels) != 1:
            raise ValueError("could not identify selected shape center")
        center = tuple(int(value) for value in black_pixels[0])
        if first_center is None:
            first_center = center
        elif center == first_center:
            break

        offsets = set()
        for action in (UP, DOWN, LEFT, RIGHT):
            moved = scout.clone()
            moved.step(action)
            after = arr(moved.frame())
            offsets.update(
                (int(row) - center[0], int(col) - center[1])
                for row, col in zip(*((before != after).nonzero()))
                if int(before[row, col]) == shape_color
                and int(after[row, col]) == background
            )
        if not offsets:
            raise ValueError("selected shape did not move")

        row_span = max(row for row, _ in offsets) - min(row for row, _ in offsets)
        col_span = max(col for _, col in offsets) - min(col for _, col in offsets)
        if row_span == 0:
            left = -min(col for _, col in offsets)
            right = max(col for _, col in offsets)
            shape = ("horizontal", left, right)
        elif col_span == 0:
            up = -min(row for row, _ in offsets)
            down = max(row for row, _ in offsets)
            shape = ("vertical", up, down)
        else:
            x_score = sum(abs(row) == abs(col) for row, col in offsets)
            diamond_radius = max(abs(row) + abs(col) for row, col in offsets)
            diamond_score = sum(
                abs(row) + abs(col) == diamond_radius
                for row, col in offsets
            )
            if x_score >= diamond_score:
                shape = (
                    "x",
                    max(max(abs(row), abs(col)) for row, col in offsets),
                )
            else:
                shape = ("diamond", diamond_radius)
        shapes.append((center, shape))
        scout.step(USE)

    max_row = frame.shape[0] - 2
    max_col = frame.shape[1] - 1

    def placements(center, shape):
        kind = shape[0]
        if kind == "horizontal":
            row_bounds = (0, max_row)
            col_bounds = (shape[1], max_col - shape[2])
            covers = lambda point, row, col: (
                point[0] == row and -shape[1] <= point[1] - col <= shape[2]
            )
        elif kind == "vertical":
            row_bounds = (shape[1], max_row - shape[2])
            col_bounds = (0, max_col)
            covers = lambda point, row, col: (
                point[1] == col and -shape[1] <= point[0] - row <= shape[2]
            )
        elif kind == "x":
            radius = shape[1]
            row_bounds = (radius, max_row - radius)
            col_bounds = (radius, max_col - radius)
            covers = lambda point, row, col: (
                abs(point[0] - row) == abs(point[1] - col)
                and abs(point[0] - row) <= radius
            )
        else:
            radius = shape[1]
            row_bounds = (radius, max_row - radius)
            col_bounds = (radius, max_col - radius)
            covers = lambda point, row, col: (
                abs(point[0] - row) + abs(point[1] - col) == radius
            )

        out = []
        for row in range(row_bounds[0], row_bounds[1] + 1):
            if (row - center[0]) % step:
                continue
            for col in range(col_bounds[0], col_bounds[1] + 1):
                if (col - center[1]) % step:
                    continue
                covered = {
                    point for point in targets if covers(point, row, col)
                }
                if covered:
                    distance = abs(row - center[0]) + abs(col - center[1])
                    out.append(((row, col), covered, distance))
        return out

    choices = [placements(center, shape) for center, shape in shapes]
    best = None

    def search(index, selected, covered, distance):
        nonlocal best
        if best is not None and distance >= best[0]:
            return
        if index == len(choices):
            if covered == targets:
                best = (distance, list(selected))
            return
        for target, newly_covered, cost in choices[index]:
            selected.append(target)
            search(
                index + 1,
                selected,
                covered | newly_covered,
                distance + cost,
            )
            selected.pop()

    search(0, [], set(), 0)
    if best is None:
        raise ValueError("no exact ring cover found")

    for index, ((current, _), target) in enumerate(zip(shapes, best[1])):
        if index:
            env.step(USE)
        _move_on_lattice(
            env,
            target[0] - current[0],
            target[1] - current[1],
            step,
        )


def cover_colored_ring_groups_with_selected_shapes(env, ring_color=4):
    """Place each selectable shape over the geometrically compatible ring group."""
    frame = arr(env.frame())
    rings = [
        blob
        for blob in connected_components(frame, colors=(ring_color,), min_area=8)
        if blob.size == (3, 3) and blob.area == 8
    ]
    if not rings:
        raise ValueError("no ring markers found")

    groups = defaultdict(set)
    for ring in rings:
        center = (ring.bbox[0] + 1, ring.bbox[1] + 1)
        groups[int(frame[center])].add(center)
    if len(groups) < 2:
        raise ValueError("colored-group leg requires multiple marker colors")

    background = _most_common(int(value) for value in frame.flat)
    step = rings[0].size[0]
    shapes = []
    scout = env.clone()
    first_center = None
    while True:
        before = arr(scout.frame()).copy()
        black_pixels = list(zip(*((before == 0).nonzero())))
        if len(black_pixels) != 1:
            raise ValueError("could not identify selected shape center")
        center = tuple(int(value) for value in black_pixels[0])
        if first_center is None:
            first_center = center
        elif center == first_center:
            break

        erased = []
        for action in (UP, DOWN, LEFT, RIGHT):
            moved = scout.clone()
            moved.step(action)
            after = arr(moved.frame())
            erased.extend(
                (int(row), int(col), int(before[row, col]))
                for row, col in zip(*((before != after).nonzero()))
                if int(after[row, col]) == background
                and int(before[row, col]) not in (0, background, ring_color)
            )
        if not erased:
            raise ValueError("selected shape did not move")
        shape_color = _most_common(color for _, _, color in erased)
        offsets = {
            (row - center[0], col - center[1])
            for row, col, color in erased
            if color == shape_color
        }
        offsets.add((0, 0))
        radius = max(max(abs(row), abs(col)) for row, col in offsets)
        scores = {
            "plus": sum(row == 0 or col == 0 for row, col in offsets),
            "x": sum(abs(row) == abs(col) for row, col in offsets),
            "diamond": sum(
                abs(row) + abs(col) == radius for row, col in offsets
            ),
        }
        kind = max(scores, key=scores.get)
        shapes.append((center, (kind, radius)))
        scout.step(USE)

    choices = []
    for center, shape in shapes:
        kind, radius = shape

        def covers(point, candidate):
            row = point[0] - candidate[0]
            col = point[1] - candidate[1]
            if kind == "plus":
                return (row == 0 or col == 0) and max(abs(row), abs(col)) <= radius
            if kind == "x":
                return abs(row) == abs(col) and abs(row) <= radius
            return abs(row) + abs(col) == radius

        shape_choices = []
        for group_color, targets in groups.items():
            candidates = [
                (row, col)
                for row in range(radius, frame.shape[0] - radius)
                for col in range(radius, frame.shape[1] - radius)
                if (row - center[0]) % step == 0
                and (col - center[1]) % step == 0
                and all(covers(target, (row, col)) for target in targets)
            ]
            if candidates:
                target = min(
                    candidates,
                    key=lambda point: (
                        abs(point[0] - center[0]) + abs(point[1] - center[1]),
                        point,
                    ),
                )
                shape_choices.append((
                    group_color,
                    target,
                    abs(target[0] - center[0]) + abs(target[1] - center[1]),
                ))
        choices.append(shape_choices)

    best = None

    def assign(index, used_groups, placements, distance):
        nonlocal best
        if best is not None and distance >= best[0]:
            return
        if index == len(shapes):
            if used_groups == set(groups):
                best = (distance, list(placements))
            return
        for group_color, target, cost in choices[index]:
            if group_color in used_groups:
                continue
            placements.append(target)
            assign(
                index + 1,
                used_groups | {group_color},
                placements,
                distance + cost,
            )
            placements.pop()

    assign(0, set(), [], 0)
    if best is None:
        raise ValueError("no one-to-one shape/ring-group cover found")

    for index, ((current, _), target) in enumerate(zip(shapes, best[1])):
        if index:
            env.step(USE)
        _move_on_lattice(
            env,
            target[0] - current[0],
            target[1] - current[1],
            step,
        )
