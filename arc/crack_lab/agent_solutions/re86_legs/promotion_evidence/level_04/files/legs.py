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


def repaint_selected_shapes_to_cover_colored_ring_markers(
        env, ring_color=4, station_border_color=2):
    """Repaint selectable shapes at swatches, then cover same-colour markers."""
    frame = arr(env.frame())
    background = _most_common(int(value) for value in frame.flat)
    rings = [
        blob
        for blob in connected_components(frame, colors=(ring_color,), min_area=8)
        if blob.size == (3, 3) and blob.area == 8
    ]
    if not rings:
        raise ValueError("no ring markers found")

    step = rings[0].size[0]
    targets = defaultdict(set)
    for ring in rings:
        point = (ring.bbox[0] + 1, ring.bbox[1] + 1)
        targets[int(frame[point])].add(point)
    all_targets = set().union(*targets.values())

    stations = {}
    for border in connected_components(
            frame, colors=(station_border_color,), min_area=20):
        if border.size != (6, 6) or border.area != 20:
            continue
        r0, c0, r1, c1 = border.bbox
        interior = frame[r0 + 1:r1, c0 + 1:c1]
        colors = {
            int(value) for value in interior.flat
            if int(value) not in (background, station_border_color)
        }
        if len(colors) == 1:
            stations[colors.pop()] = border.bbox

    scout = env.clone()
    shapes = []
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

        color_votes = []
        moved_frames = []
        for action in (UP, DOWN, LEFT, RIGHT):
            moved = scout.clone()
            moved.step(action)
            after = arr(moved.frame())
            moved_frames.append(after)
            color_votes.extend(
                int(before[row, col])
                for row, col in zip(*((before != after).nonzero()))
                if int(after[row, col]) == background
                and int(before[row, col]) not in (
                    0, background, ring_color, station_border_color)
            )
        if not color_votes:
            raise ValueError("selected shape did not move")
        shape_color = _most_common(color_votes)
        offsets = {
            (int(row) - center[0], int(col) - center[1])
            for after in moved_frames
            for row, col in zip(*((before != after).nonzero()))
            if int(before[row, col]) == shape_color
            and int(after[row, col]) == background
        }
        if not offsets:
            raise ValueError("could not recover selected shape geometry")

        axis_score = sum(row == 0 or col == 0 for row, col in offsets)
        x_score = sum(abs(row) == abs(col) for row, col in offsets)
        if axis_score >= len(offsets) * 3 // 4:
            row_span = max(row for row, _ in offsets) - min(
                row for row, _ in offsets)
            col_span = max(col for _, col in offsets) - min(
                col for _, col in offsets)
            if row_span and col_span:
                shape = (
                    "plus",
                    max(max(abs(row), abs(col)) for row, col in offsets),
                )
            elif row_span:
                shape = (
                    "vertical",
                    -min(row for row, _ in offsets),
                    max(row for row, _ in offsets),
                )
            else:
                shape = (
                    "horizontal",
                    -min(col for _, col in offsets),
                    max(col for _, col in offsets),
                )
        elif x_score >= len(offsets) * 3 // 4:
            shape = (
                "x",
                max(max(abs(row), abs(col)) for row, col in offsets),
            )
        else:
            shape = (
                "diamond",
                max(abs(row) + abs(col) for row, col in offsets),
            )
        shapes.append((center, shape_color, shape))
        scout.step(USE)

    def station_point(bbox, center):
        r0, c0, r1, c1 = bbox
        rows = [
            row for row in range(r0 + 1, r1)
            if (row - center[0]) % step == 0
        ]
        cols = [
            col for col in range(c0 + 1, c1)
            if (col - center[1]) % step == 0
        ]
        if not rows or not cols:
            raise ValueError("paint station is off the movement lattice")
        return rows[0], cols[0]

    def covers(shape, point, row, col):
        kind = shape[0]
        dr, dc = point[0] - row, point[1] - col
        if kind == "plus":
            return (dr == 0 or dc == 0) and max(abs(dr), abs(dc)) <= shape[1]
        if kind == "x":
            return abs(dr) == abs(dc) and abs(dr) <= shape[1]
        if kind == "vertical":
            return dc == 0 and -shape[1] <= dr <= shape[2]
        if kind == "horizontal":
            return dr == 0 and -shape[1] <= dc <= shape[2]
        return abs(dr) + abs(dc) == shape[1]

    def fits(shape, row, col):
        kind = shape[0]
        if kind in ("plus", "x", "diamond"):
            up = down = left = right = shape[1]
        elif kind == "vertical":
            up, down, left, right = shape[1], shape[2], 0, 0
        else:
            up, down, left, right = 0, 0, shape[1], shape[2]
        return (
            up <= row < frame.shape[0] - down - 1
            and left <= col < frame.shape[1] - right
        )

    choices = []
    for center, _, shape in shapes:
        shape_choices = []
        for color, group in targets.items():
            if color not in stations:
                continue
            paint = station_point(stations[color], center)
            for row in range(center[0] % step, frame.shape[0] - 1, step):
                for col in range(center[1] % step, frame.shape[1], step):
                    if not fits(shape, row, col):
                        continue
                    covered = {
                        point for point in all_targets
                        if covers(shape, point, row, col)
                    }
                    if covered != group:
                        continue
                    cost = (
                        abs(center[0] - paint[0])
                        + abs(center[1] - paint[1])
                        + abs(paint[0] - row)
                        + abs(paint[1] - col)
                    )
                    shape_choices.append((cost, color, paint, (row, col)))
        if not shape_choices:
            raise ValueError("no painted marker cover for selected shape")
        choices.append(shape_choices)

    best = None

    def search(index, used_colors, selected, cost):
        nonlocal best
        if best is not None and cost >= best[0]:
            return
        if index == len(choices):
            if used_colors == set(targets):
                best = (cost, list(selected))
            return
        for choice in choices[index]:
            choice_cost, color, _, _ = choice
            if color in used_colors:
                continue
            selected.append(choice)
            search(
                index + 1,
                used_colors | {color},
                selected,
                cost + choice_cost,
            )
            selected.pop()

    search(0, set(), [], 0)
    if best is None:
        raise ValueError("could not assign shapes to marker colours")

    for index, ((center, _, _), (_, _, paint, target)) in enumerate(
            zip(shapes, best[1])):
        if index:
            env.step(USE)
        _move_on_lattice(
            env, paint[0] - center[0], paint[1] - center[1], step)
        _move_on_lattice(
            env, target[0] - paint[0], target[1] - paint[1], step)
