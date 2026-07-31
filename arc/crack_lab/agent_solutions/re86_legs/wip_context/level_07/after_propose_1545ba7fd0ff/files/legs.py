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


def _selected_center(env):
    frame = arr(env.frame())
    points = list(zip(*((frame == 0).nonzero())))
    if len(points) != 1:
        raise ValueError("no unique selection cursor")
    return int(points[0][0]), int(points[0][1])


def _selection_cycle(env, limit=12):
    """Centers of every selectable object, in the order USE cycles them."""
    scout = env.clone()
    centers = []
    first = None
    while True:
        center = _selected_center(scout)
        if first is None:
            first = center
        elif center == first:
            return centers
        centers.append(center)
        if len(centers) > limit:
            raise ValueError("selection cycle did not close")
        scout.step(USE)


def _park_all(env, direction, count, distance):
    """Push every selectable object off the board and return the bare frame."""
    scout = env.clone()
    for index in range(count):
        if index:
            scout.step(USE)
        for _ in range(distance):
            scout.step(direction)
    return arr(scout.frame())


def _survey_markers_and_swatches(env, count, distance, ring_color,
                                 border_color):
    """Ring markers and paint swatches, seen with the movable shapes parked."""
    markers, swatches = {}, {}
    for direction in (UP, DOWN, LEFT, RIGHT):
        frame = _park_all(env, direction, count, distance)
        background = _most_common(int(value) for value in frame.flat)
        for blob in connected_components(frame, colors=(ring_color,), min_area=8):
            if blob.size != (3, 3) or blob.area != 8:
                continue
            point = (blob.bbox[0] + 1, blob.bbox[1] + 1)
            value = int(frame[point])
            if value not in (ring_color, background):
                markers[point] = value
        for blob in connected_components(frame, colors=(border_color,), min_area=12):
            r0, c0, r1, c1 = blob.bbox
            if r1 - r0 < 3 or c1 - c0 < 3:
                continue
            cells = {
                (row, col)
                for row in range(r0 + 1, r1)
                for col in range(c0 + 1, c1)
                if int(frame[row, col]) not in (background, border_color)
            }
            values = {int(frame[point]) for point in cells}
            if len(values) == 1:
                swatches[values.pop()] = cells
    return markers, swatches


def _shape_offsets(env, index, count, distance):
    """Cell offsets of one selectable shape, with the others parked away."""
    scout = env.clone()
    for other in range(count):
        if other != index:
            for _ in range(distance):
                scout.step(UP)
        scout.step(USE)
    for _ in range(index):
        scout.step(USE)

    center = _selected_center(scout)
    before = arr(scout.frame()).copy()
    background = _most_common(int(value) for value in before.flat)
    frames = []
    votes = []
    for action in (UP, DOWN, LEFT, RIGHT):
        moved = scout.clone()
        moved.step(action)
        after = arr(moved.frame())
        frames.append(after)
        votes.extend(
            int(before[row, col])
            for row, col in zip(*((before != after).nonzero()))
            if int(after[row, col]) == background
            and int(before[row, col]) not in (0, background)
        )
    if not votes:
        raise ValueError("selected shape did not move")
    shape_color = _most_common(votes)

    # A shape cell need not vacate to the background: it may uncover a marker
    # it was hiding, so track every cell that stops being the shape colour.
    offsets = set()
    for after in frames:
        offsets.update(
            (int(row) - center[0], int(col) - center[1])
            for row, col in zip(*((before != after).nonzero()))
            if int(before[row, col]) == shape_color
        )
    offsets.discard((0, 0))

    rows, cols = before.shape[:2]
    complete = set(offsets)
    for row_offset, col_offset in list(offsets):
        for mirror in ((-row_offset, col_offset), (row_offset, -col_offset),
                       (-row_offset, -col_offset)):
            row, col = center[0] + mirror[0], center[1] + mirror[1]
            if not (0 <= row < rows and 0 <= col < cols):
                complete.add(mirror)
    return center, shape_color, complete


def _swatch_hit(offsets, center, swatches):
    cells = {(center[0] + dr, center[1] + dc) for dr, dc in offsets}
    hits = [color for color, swatch in swatches.items()
            if not swatch.isdisjoint(cells)]
    if len(hits) == 1:
        return hits[0]
    return "blocked" if hits else None


def _paint_routes(offsets, start, start_color, swatches, step, rows, cols):
    """BFS over (center, paint colour) states on the movement lattice."""
    hits = {
        (row, col): _swatch_hit(offsets, (row, col), swatches)
        for row in range(start[0] % step, rows, step)
        for col in range(start[1] % step, cols, step)
    }
    previous = {(start, start_color): None}
    queue = deque([(start, start_color)])
    while queue:
        state = queue.popleft()
        (row, col), color = state
        for dr, dc in ((-step, 0), (step, 0), (0, -step), (0, step)):
            point = (row + dr, col + dc)
            hit = hits.get(point, "blocked")
            if hit == "blocked":
                continue
            following = (point, color if hit is None else hit)
            if following in previous:
                continue
            previous[following] = state
            queue.append(following)
    return previous


def _route_centers(previous, state):
    path = []
    while state is not None:
        path.append(state[0])
        state = previous[state]
    return path[::-1]


def paint_shapes_at_swatches_to_cover_ring_markers(
        env, ring_color=4, station_border_color=2):
    """Cover every ring marker with a shape recoloured by a paint swatch.

    Unlike the earlier cover legs, a shape is repainted whenever ANY of its
    cells touches a swatch, so the route to a placement is searched too, not
    just the placement itself.
    """
    frame = arr(env.frame())
    rows, cols = frame.shape[:2]
    rings = [
        blob
        for blob in connected_components(frame, colors=(ring_color,), min_area=8)
        if blob.size == (3, 3) and blob.area == 8
    ]
    if not rings:
        raise ValueError("no ring markers found")
    step = rings[0].size[0]
    starts = _selection_cycle(env)
    distance = max(rows, cols) // step + 1

    markers, swatches = _survey_markers_and_swatches(
        env, len(starts), distance, ring_color, station_border_color)
    if not markers or not swatches:
        raise ValueError("no markers or paint swatches found")
    remaining = set(markers)

    plans = []
    for index in range(len(starts)):
        center, start_color, offsets = _shape_offsets(
            env, index, len(starts), distance)
        previous = _paint_routes(
            offsets, center, start_color, swatches, step, rows - 1, cols)
        best = {}
        for point, color in previous:
            covered = frozenset(
                marker for marker in markers
                if (marker[0] - point[0], marker[1] - point[1]) in offsets
            )
            if not covered or {markers[m] for m in covered} != {color}:
                continue
            path = _route_centers(previous, (point, color))
            if covered not in best or len(path) < len(best[covered]):
                best[covered] = path
        if not best:
            raise ValueError(f"shape {index} cannot cover a marker group")
        plans.append(sorted(best.items(), key=lambda item: len(item[1])))

    chosen = None

    def search(index, covered, picked, cost):
        nonlocal chosen
        if chosen is not None and cost >= chosen[0]:
            return
        if index == len(plans):
            if covered == remaining:
                chosen = (cost, list(picked))
            return
        for group, path in plans[index]:
            picked.append(path)
            search(index + 1, covered | group, picked, cost + len(path))
            picked.pop()

    search(0, set(), [], 0)
    if chosen is None:
        raise ValueError("no painted placement covers every marker")

    for index, path in enumerate(chosen[1]):
        if index:
            env.step(USE)
        for current, following in zip(path, path[1:]):
            _move_on_lattice(
                env, following[0] - current[0], following[1] - current[1], step)


def deform_selected_shapes_to_colored_ring_axes(env, ring_color=4):
    """Use barriers to reshape an outline and cross onto their coloured rings."""
    frame = arr(env.frame())
    rings = [
        blob
        for blob in connected_components(frame, colors=(ring_color,), min_area=8)
        if blob.size == (3, 3) and blob.area == 8
    ]
    if len(rings) != 8:
        raise ValueError("expected eight coloured ring markers")

    targets = defaultdict(list)
    for ring in rings:
        center = (ring.bbox[0] + 1, ring.bbox[1] + 1)
        targets[int(frame[center])].append(center)

    rectangle_groups = []
    axis_groups = []
    for color, points in targets.items():
        row_counts = Counter(row for row, _ in points)
        col_counts = Counter(col for _, col in points)
        if len(points) != 4:
            continue
        if len(row_counts) == 2 and len(col_counts) == 2:
            rectangle_groups.append((color, points))
        elif (
            len(row_counts) == 3
            and len(col_counts) == 3
            and max(row_counts.values()) == 2
            and max(col_counts.values()) == 2
        ):
            axis_groups.append((color, points))
    if len(rectangle_groups) != 1 or len(axis_groups) != 1:
        raise ValueError("could not identify rectangle and cross ring targets")

    step = rings[0].size[0]
    _, rectangle = rectangle_groups[0]
    rectangle_rows = sorted({row for row, _ in rectangle})
    rectangle_cols = sorted({col for _, col in rectangle})
    _, axes = axis_groups[0]
    axis_rows = Counter(row for row, _ in axes)
    axis_cols = Counter(col for _, col in axes)
    target_row = next(row for row, count in axis_rows.items() if count == 2)
    target_col = next(col for col, count in axis_cols.items() if count == 2)

    signature = (
        rectangle_rows[1] - rectangle_rows[0],
        rectangle_cols[1] - rectangle_cols[0],
        sorted(row - target_row for row, col in axes if col == target_col),
        sorted(col - target_col for row, col in axes if row == target_row),
    )
    expected = (
        9 * step,
        3 * step,
        [-step, 6 * step],
        [-step, 6 * step],
    )
    if signature != expected:
        raise ValueError("unsupported deformable ring geometry")

    square_maneuvers = (
        (-3 * step, 0),
        (0, 4 * step),
        (-8 * step, 0),
        (0, 9 * step),
        (10 * step, 0),
    )
    cross_maneuvers = (
        (-2 * step, 0),
        (0, -7 * step),
        (3 * step, 0),
        (0, 3 * step),
        (-3 * step, 0),
        (0, -5 * step),
        (8 * step, 0),
        (-6 * step, 0),
    )

    def execute(node):
        for row_delta, col_delta in square_maneuvers:
            _move_on_lattice(node, row_delta, col_delta, step)
        node.step(USE)
        for row_delta, col_delta in cross_maneuvers:
            _move_on_lattice(node, row_delta, col_delta, step)

    base_level = env.levels_completed
    scout = env.clone()
    execute(scout)
    if scout.levels_completed <= base_level:
        raise ValueError("deformation route did not cover all ring axes")
    execute(env)


def deform_and_repaint_selected_shapes_to_colored_markers(
    env, ring_color=4
):
    """Reshape three selected movers, repaint them, and cover their markers."""
    frame = arr(env.frame())
    rings = [
        blob
        for blob in connected_components(
            frame, colors=(ring_color,), min_area=8
        )
        if blob.size == (3, 3) and blob.area == 8
    ]
    if len(rings) != 9:
        raise ValueError("expected nine coloured ring markers")

    targets = defaultdict(list)
    for ring in rings:
        center = (ring.bbox[0] + 1, ring.bbox[1] + 1)
        targets[int(frame[center])].append(center)

    step = rings[0].size[0]

    def normalized(points):
        row0 = min(row for row, _ in points)
        col0 = min(col for _, col in points)
        return sorted(
            ((row - row0) // step, (col - col0) // step)
            for row, col in points
        )

    signature = {
        color: normalized(points)
        for color, points in targets.items()
    }
    expected = {
        8: [(0, 2), (2, 0), (2, 11), (6, 2)],
        9: [(0, 6), (2, 0)],
        11: [(0, 2), (6, 0), (6, 4)],
    }
    if signature != expected:
        raise ValueError("unsupported deform-and-repaint marker geometry")

    small_cross = (
        (0, 10 * step),
        (-7 * step, 0),
        (0, step),
        (4 * step, 0),
        (-9 * step, 0),
        (0, -6 * step),
        (2 * step, 0),
        (0, 6 * step),
        (7 * step, 0),
    )
    outline = (
        (-3 * step, 0),
        (0, 4 * step),
        (-2 * step, 0),
        (0, 5 * step),
        (-6 * step, 0),
        (0, -13 * step),
        (-5 * step, 0),
        (5 * step, 0),
        (0, 13 * step),
    )
    large_cross_base = (
        (-step, 0),
        (0, -2 * step),
        (-8 * step, 0),
        (0, -3 * step),
        (-3 * step, 0),
        (0, 3 * step),
        (4 * step, 0),
        (-4 * step, 0),
        (0, step),
        (-3 * step, 0),
        (3 * step, 0),
        (0, 2 * step),
        (step, 0),
        (0, step),
        (0, -step),
        (-step, 0),
        (0, -3 * step),
        (-step, 0),
        (step, 0),
        (0, -2 * step),
    )
    large_cross_shift = (
        (0, 5 * step),
        (step, 0),
        (0, 2 * step),
        (0, -2 * step),
        (-step, 0),
        (0, -4 * step),
    )
    large_cross_paint = (
        (-4 * step, 0),
        (4 * step, 0),
    )

    def follow(node, maneuvers):
        for row_delta, col_delta in maneuvers:
            _move_on_lattice(node, row_delta, col_delta, step)

    def execute(node):
        follow(node, small_cross)
        node.step(USE)
        follow(node, outline)
        node.step(USE)
        follow(node, large_cross_base)
        follow(node, large_cross_shift)
        follow(node, large_cross_shift)
        follow(node, large_cross_paint)

    base_level = env.levels_completed
    scout = env.clone()
    execute(scout)
    if scout.levels_completed <= base_level:
        raise ValueError(
            "deform-and-repaint route did not cover all coloured markers"
        )
    execute(env)
