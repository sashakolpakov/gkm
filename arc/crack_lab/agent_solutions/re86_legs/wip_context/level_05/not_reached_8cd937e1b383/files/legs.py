# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.

from collections import Counter, defaultdict, deque

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


_STEP_ACTIONS = ((UP, (-1, 0)), (DOWN, (1, 0)), (LEFT, (0, -1)), (RIGHT, (0, 1)))
_PROBE_ACTIONS = (UP, DOWN, LEFT, RIGHT)


def _selected_center(frame):
    black = list(zip(*((frame == 0).nonzero())))
    if len(black) != 1:
        raise ValueError("could not identify the selected shape")
    return tuple(int(value) for value in black[0])


def _ring_markers(frame, ring_color):
    """Map each 3x3 ring marker's centre to the colour painted inside it."""
    markers = {}
    for blob in connected_components(frame, colors=(ring_color,), min_area=8):
        if blob.size == (3, 3) and blob.area == 8:
            center = (blob.bbox[0] + 1, blob.bbox[1] + 1)
            markers[center] = int(frame[center])
    return markers


def _classify_shape(offsets):
    """Name the selected shape's geometry from the cells it vacates."""
    radius = max(max(abs(row), abs(col)) for row, col in offsets)
    if all(row == 0 or col == 0 for row, col in offsets):
        rows = {row for row, _ in offsets}
        cols = {col for _, col in offsets}
        if len(rows) > 1 and len(cols) > 1:
            return ("plus", radius)
        if len(rows) > 1:
            return ("vertical", -min(rows), max(rows))
        return ("horizontal", -min(cols), max(cols))
    if all(abs(row) == abs(col) for row, col in offsets):
        return ("x", radius)
    return ("diamond", max(abs(row) + abs(col) for row, col in offsets))


def _shape_cells(shape):
    """The shape's ideal cell offsets, refilling anything occlusion hid."""
    kind = shape[0]
    if kind == "plus":
        radius = shape[1]
        spokes = {(sign * step, 0) for step in range(1, radius + 1) for sign in (1, -1)}
        spokes.update((0, sign * step) for step in range(1, radius + 1) for sign in (1, -1))
        return {(0, 0)} | spokes
    if kind == "x":
        radius = shape[1]
        return {(row_sign * step, col_sign * step)
                for step in range(0, radius + 1)
                for row_sign in (1, -1) for col_sign in (1, -1)}
    if kind == "diamond":
        radius = shape[1]
        return {(row, col)
                for row in range(-radius, radius + 1)
                for col in range(-radius, radius + 1)
                if abs(row) + abs(col) == radius}
    if kind == "vertical":
        return {(row, 0) for row in range(-shape[1], shape[2] + 1)}
    return {(0, col) for col in range(-shape[1], shape[2] + 1)}


def _selected_shape(node, background, ignore):
    """(centre, colour, geometry) of the shape currently selected on ``node``."""
    before = arr(node.frame()).copy()
    center = _selected_center(before)
    votes = []
    moved_frames = []
    for action in _PROBE_ACTIONS:
        moved = node.clone()
        moved.step(action)
        after = arr(moved.frame())
        moved_frames.append(after)
        votes.extend(
            int(before[row, col])
            for row, col in zip(*((before != after).nonzero()))
            if int(after[row, col]) == background
            and int(before[row, col]) not in ignore
        )
    if not votes:
        raise ValueError("the selected shape did not move")
    color = _most_common(votes)
    offsets = {
        (int(row) - center[0], int(col) - center[1])
        for after in moved_frames
        for row, col in zip(*((before != after).nonzero()))
        if int(before[row, col]) == color and int(after[row, col]) == background
    }
    if not offsets:
        raise ValueError("could not recover the selected shape's geometry")
    return center, color, _classify_shape(offsets)


def _shape_cycle(env, background, ignore):
    """Walk the selection cycle once, describing every selectable shape."""
    scout = env.clone()
    shapes = []
    first = None
    while True:
        center, color, shape = _selected_shape(scout, background, ignore)
        if first is None:
            first = center
        elif center == first:
            return shapes
        shapes.append((center, color, shape))
        scout.step(USE)


def _uncovered_ring_markers(env, ring_color, shape_count, probe_steps=(1, 3)):
    """Ring markers, including ones a shape is currently sitting on top of.

    A shape is drawn over the markers it covers, which breaks their rings in the
    frame. Nudging each shape on a clone exposes whatever it was hiding.
    """
    markers = dict(_ring_markers(arr(env.frame()), ring_color))
    for index in range(shape_count):
        for action in _PROBE_ACTIONS:
            for repeats in probe_steps:
                scout = env.clone()
                for _ in range(index):
                    scout.step(USE)
                for _ in range(repeats):
                    scout.step(action)
                for center, color in _ring_markers(arr(scout.frame()), ring_color).items():
                    markers.setdefault(center, color)
    return markers


def _paint_stations(frame, background, border_color):
    """Map each swatch colour to the 6x6 station box that paints it."""
    stations = {}
    for border in connected_components(frame, colors=(border_color,), min_area=20):
        if border.size != (6, 6) or border.area != 20:
            continue
        r0, c0, r1, c1 = border.bbox
        interior = frame[r0 + 1:r1, c0 + 1:c1]
        colors = {
            int(value) for value in interior.flat
            if int(value) not in (background, border_color)
        }
        if len(colors) == 1:
            stations[colors.pop()] = border.bbox
    return stations


def _route(start, goal, step, bounds, blocked):
    """Shortest lattice walk from ``start`` to ``goal`` avoiding ``blocked``."""
    if start == goal:
        return []
    rows, cols = bounds
    came = {start: None}
    queue = deque([start])
    while queue:
        point = queue.popleft()
        for action, (row_delta, col_delta) in _STEP_ACTIONS:
            nxt = (point[0] + row_delta * step, point[1] + col_delta * step)
            if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols) or nxt in came:
                continue
            if nxt in blocked and nxt != goal:
                continue
            came[nxt] = (point, action)
            if nxt == goal:
                path = []
                cursor = nxt
                while came[cursor] is not None:
                    previous, taken = came[cursor]
                    path.append(taken)
                    cursor = previous
                return path[::-1]
            queue.append(nxt)
    raise ValueError("no swatch-free route on the movement lattice")


def repaint_shape_set_to_cover_colored_ring_markers(
        env, ring_color=4, station_border_color=2):
    """Repaint and place the selectable shapes so they jointly cover every marker.

    Generalises the one-shape-per-colour version: shapes may share a colour and
    split that colour's markers between them, so the assignment is a set cover
    rather than a bijection. Markers hidden under a shape are recovered first,
    and each walk is routed around the other swatches so a shape is not
    repainted in transit. The plan is checked on a clone before it is committed.
    """
    frame = arr(env.frame())
    background = _most_common(int(value) for value in frame.flat)
    ignore = (0, background, ring_color, station_border_color)

    shapes = _shape_cycle(env, background, ignore)
    if not shapes:
        raise ValueError("no selectable shapes found")

    markers = _uncovered_ring_markers(env, ring_color, len(shapes))
    if not markers:
        raise ValueError("no ring markers found")
    marker_set = frozenset(markers)

    stations = _paint_stations(frame, background, station_border_color)
    rings = [
        blob
        for blob in connected_components(frame, colors=(ring_color,), min_area=8)
        if blob.size == (3, 3) and blob.area == 8
    ]
    step = rings[0].size[0] if rings else 3
    bounds = (frame.shape[0] - 1, frame.shape[1])
    interiors = {
        (row, col)
        for r0, c0, r1, c1 in stations.values()
        for row in range(r0 + 1, r1)
        for col in range(c0 + 1, c1)
    }

    def swatch_point(bbox, center):
        r0, c0, r1, c1 = bbox
        points = [
            (row, col)
            for row in range(r0 + 1, r1) if (row - center[0]) % step == 0
            for col in range(c0 + 1, c1) if (col - center[1]) % step == 0
        ]
        if not points:
            raise ValueError("a paint swatch is off the movement lattice")
        return points[0]

    def covered(cells, center):
        return frozenset(
            point for point in marker_set
            if (point[0] - center[0], point[1] - center[1]) in cells
        )

    def options(index, cells, strict):
        """Placements for one shape: (cost, colour, swatch point, target, gains)."""
        center, color, _ = shapes[index]
        out = {}
        for target_color in set(markers.values()) | {color}:
            if target_color != color and target_color not in stations:
                continue
            swatch = (
                None if target_color == color
                else swatch_point(stations[target_color], center)
            )
            for row in range(center[0] % step, bounds[0], step):
                for col in range(center[1] % step, bounds[1], step):
                    if (row, col) in interiors and (row, col) != swatch:
                        continue
                    hits = covered(cells, (row, col))
                    gains = frozenset(
                        point for point in hits
                        if markers[point] == target_color
                    )
                    if strict and hits != gains:
                        continue
                    if not gains and (row, col) != center:
                        continue
                    cost = abs(row - center[0]) + abs(col - center[1])
                    if swatch is not None:
                        cost = (
                            abs(swatch[0] - center[0]) + abs(swatch[1] - center[1])
                            + abs(row - swatch[0]) + abs(col - swatch[1])
                        )
                    choice = (cost, target_color, swatch, (row, col), gains)
                    if gains not in out or choice < out[gains]:
                        out[gains] = choice
        return sorted(out.values())

    def assign(choices):
        best = [None]

        def search(index, gained, picked, cost):
            if best[0] is not None and cost >= best[0][0]:
                return
            if index == len(choices):
                if gained == marker_set:
                    best[0] = (cost, list(picked))
                return
            remaining = frozenset().union(
                *([frozenset()] + [
                    frozenset().union(*([frozenset()] + [c[4] for c in choices[later]]))
                    for later in range(index, len(choices))
                ])
            )
            if not marker_set <= gained | remaining:
                return
            for choice in choices[index]:
                picked.append(choice)
                search(index + 1, gained | choice[4], picked, cost + choice[0])
                picked.pop()

        search(0, frozenset(), [], 0)
        return best[0]

    def moves_for(picked):
        moves = []
        for index, (_, _, swatch, target, _) in enumerate(picked):
            if index:
                moves.append(USE)
            center = shapes[index][0]
            if swatch is not None:
                moves.extend(_route(center, swatch, step, bounds, interiors))
                center = swatch
            moves.extend(_route(center, target, step, bounds, interiors))
        return moves

    def replay(moves):
        scout = env.clone()
        for action in moves:
            scout.step(action)
        return scout.levels_completed > env.levels_completed

    base_cells = [_shape_cells(shape) for _, _, shape in shapes]
    for toggle_center in (False, True):
        cell_sets = [
            cells ^ {(0, 0)} if toggle_center else cells
            for cells in base_cells
        ]
        for strict in (True, False):
            plan = assign([
                options(index, cell_sets[index], strict)
                for index in range(len(shapes))
            ])
            if plan is None:
                continue
            moves = moves_for(plan[1])
            if replay(moves):
                for action in moves:
                    env.step(action)
                return
    raise ValueError("no marker cover found for the selectable shapes")
