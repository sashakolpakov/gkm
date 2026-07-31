# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.

from collections import deque

from perception import connected_components


PROTOCOL_ROWS = {
    "D": (0, 1),
    "U": (0, 5),
    "L": (1, 5),
    "R": (1,),
    "X": (3,),
    "M": (0, 3),
    "A": (0, 2),
    "B": (0, 1, 2, 3, 4, 5),
}


def find_right_segment_panel(frame):
    """Return the segments and lattice coordinates of the right-hand panel."""
    width = len(frame[0])
    segments = [
        blob
        for blob in connected_components(frame, colors=(1, 5), min_area=3)
        if blob.area == 3
        and 1 in blob.size
        and blob.centroid[1] > width / 2
    ]
    rows = sorted({int(round(blob.centroid[0])) for blob in segments})
    cols = sorted({int(round(blob.centroid[1])) for blob in segments})
    return segments, rows, cols


def find_upper_right_components(frame, colors, min_area=4):
    """Return matching components in the upper-right play area."""
    height, width = len(frame), len(frame[0])
    return [
        blob
        for blob in connected_components(frame, colors=colors, min_area=min_area)
        if blob.bbox[2] < height / 2 and blob.bbox[1] > width / 2
    ]


def find_upper_right_board_objects(frame):
    """Return color-11 board objects in the upper-right play area."""
    return find_upper_right_components(frame, colors=(11,))


def find_upper_right_board_tiles(frame):
    """Return solid square color-4 tiles in the upper-right play area."""
    return [
        blob
        for blob in find_upper_right_components(frame, colors=(4,), min_area=9)
        if blob.size[0] == blob.size[1]
        and blob.area == blob.size[0] * blob.size[1]
    ]


def find_lattice_route(
    frame,
    start,
    destination,
    step,
    max_steps,
    direction_order,
):
    """Find a bounded route over walkable cells in the upper-right lattice."""
    height, width = len(frame), len(frame[0])
    direction_delta = {
        "L": (0, -step),
        "R": (0, step),
        "U": (-step, 0),
        "D": (step, 0),
    }

    def walkable(position):
        row, col = position
        if (
            row < 0
            or col <= width / 2
            or row + step > height / 2
            or col + step > width
        ):
            return False
        return all(
            int(frame[r][c]) not in (0, 6)
            for r in range(row, row + step)
            for c in range(col, col + step)
        )

    queue = deque([(start, [])])
    seen = {start}
    while queue:
        position, path = queue.popleft()
        if position == destination:
            return path
        if len(path) >= max_steps:
            continue
        for direction in direction_order:
            dr, dc = direction_delta[direction]
            neighbor = (position[0] + dr, position[1] + dc)
            if neighbor in seen or not walkable(neighbor):
                continue
            seen.add(neighbor)
            queue.append((neighbor, path + [direction]))
    return None


def snap_to_lattice(value, origin, step):
    """Snap one coordinate to the nearest cell on a lattice."""
    return origin + int(round((value - origin) / step)) * step


def write_protocol_to_segment_panel(
    env,
    frame,
    rows,
    cols,
    protocol,
    protocol_rows=None,
):
    """Set panel columns to the six-row glyph for each protocol symbol."""
    codes = PROTOCOL_ROWS if protocol_rows is None else protocol_rows
    for col_index, direction in enumerate(protocol):
        desired_rows = codes[direction]
        for row_index, row in enumerate(rows):
            is_on = int(frame[row][cols[col_index]]) == 5
            should_be_on = row_index in desired_rows
            if is_on != should_be_on:
                env.step(6, cols[col_index], row)
    return True


def click_largest_color_9_submit_disc(env, frame=None):
    """Click the largest two-dimensional color-9 disc."""
    if frame is None:
        frame = env.frame()
    submit_discs = [
        blob
        for blob in connected_components(frame, colors=(9,), min_area=4)
        if blob.size[0] > 1 and blob.size[1] > 1
    ]
    if submit_discs:
        disc = max(submit_discs, key=lambda blob: blob.area)
        row, col = disc.centroid
        env.step(6, int(round(col)), int(round(row)))


def configure_then_submit(env, configure, submit_captured_frame=False):
    """Run a configuration leg, then submit its result when successful."""
    configured = configure(env)
    if configured is None or configured is False:
        return False
    frame = configured if submit_captured_frame else None
    click_largest_color_9_submit_disc(env, frame)
    return True


def make_small_segments_color_5(env):
    """Toggle each small color-1 line segment."""
    segments = [
        blob
        for blob in connected_components(env.frame(), colors=(1,), min_area=3)
        if blob.area == 3 and 1 in blob.size
    ]
    for blob in segments:
        row, col = blob.centroid
        env.step(6, int(round(col)), int(round(row)))
    return True


def turn_on_outer_rows_of_right_segment_panel(env):
    """Turn on the top and bottom rows of the right segment panel."""
    frame = env.frame()
    segments, _, _ = find_right_segment_panel(frame)
    if not segments:
        return None

    rows = [blob.centroid[0] for blob in segments]
    outer_rows = (min(rows), max(rows))
    for blob in segments:
        if blob.color == 1 and any(
            abs(blob.centroid[0] - row) < 0.5 for row in outer_rows
        ):
            row, col = blob.centroid
            env.step(6, int(round(col)), int(round(row)))
    return frame


def encode_reacquisition_route_through_barrier(env):
    """Route the small cyan agent through a wall gap and encode the route."""
    frame = env.frame()
    height, width = len(frame), len(frame[0])

    _, rows, cols = find_right_segment_panel(frame)

    board_objects = find_upper_right_board_objects(frame)
    sources = [
        blob for blob in board_objects if blob.area == 14 and blob.size == (4, 4)
    ]
    targets = [blob for blob in board_objects if blob not in sources]
    if len(rows) != 6 or not cols or not sources or not targets:
        return False

    source = sources[0]
    target = max(targets, key=lambda blob: blob.area)
    step = source.size[0]
    source_row, source_col = source.top_left

    target_row = snap_to_lattice(target.centroid[0], source_row, step)
    target_col = snap_to_lattice(target.centroid[1], source_col, step)

    barriers = [
        blob
        for blob in connected_components(frame, colors=(6,), min_area=step * step)
        if blob.bbox[2] < height / 2
    ]
    route = []

    def add_axis_moves(start, end, negative, positive):
        direction = negative if end < start else positive
        route.extend(direction for _ in range(abs(end - start) // step))

    separating = []
    for barrier in barriers:
        barrier_col = snap_to_lattice(barrier.centroid[1], source_col, step)
        if min(source_col, target_col) < barrier_col < max(source_col, target_col):
            separating.append((barrier_col, barrier))

    if separating:
        wall_col = separating[0][0]
        wall_top = min(blob.bbox[0] for _, blob in separating)
        wall_bottom = max(blob.bbox[2] for _, blob in separating)
        first_lattice_row = snap_to_lattice(wall_top, source_row, step)
        gap_rows = []
        for row in range(first_lattice_row, wall_bottom + 1, step):
            wall_pixels = sum(
                int(frame[r][c]) == 6
                for r in range(row, row + step)
                for c in range(wall_col, wall_col + step)
            )
            if wall_pixels == 0:
                gap_rows.append(row)
        if not gap_rows:
            return False
        gap_row = min(
            gap_rows,
            key=lambda row: abs(source_row - row) + abs(target_row - row),
        )
        add_axis_moves(source_row, gap_row, "U", "D")
        add_axis_moves(source_col, target_col, "L", "R")
        add_axis_moves(gap_row, target_row, "U", "D")
    else:
        add_axis_moves(source_col, target_col, "L", "R")
        add_axis_moves(source_row, target_row, "U", "D")

    if len(route) != len(cols):
        return False

    write_protocol_to_segment_panel(env, frame, rows, cols, route)
    return True


def reacquire_max_scaled_agent_and_route_to_socket(env):
    """Collapse a scaled agent and route one lattice cell at a time."""
    frame = env.frame()

    _, rows, cols = find_right_segment_panel(frame)
    if len(rows) != 6 or not cols:
        return False

    board_tiles = find_upper_right_board_tiles(frame)
    if not board_tiles:
        return False
    step = min(blob.size[0] for blob in board_tiles)

    board_objects = find_upper_right_board_objects(frame)
    sources = [
        blob
        for blob in board_objects
        if blob.size[0] == blob.size[1] and blob.size[0] >= step
    ]
    targets = [blob for blob in board_objects if blob not in sources]
    if not sources or not targets:
        return False

    source = max(sources, key=lambda blob: blob.area)
    target = max(targets, key=lambda blob: blob.area)
    scale = source.size[0] // step
    reacquisitions = max(0, scale - 1)

    start = source.top_left
    destination = (target.bbox[0], target.bbox[1] + 1)

    horizontal = ("L", "R") if destination[1] < start[1] else ("R", "L")
    vertical = ("U", "D") if destination[0] < start[0] else ("D", "U")
    direction_order = horizontal + vertical

    route = find_lattice_route(
        frame,
        start,
        destination,
        step,
        len(cols) - reacquisitions,
        direction_order,
    )

    protocol = ["M"] * reacquisitions + (route or [])
    if route is None or len(protocol) != len(cols):
        return False

    write_protocol_to_segment_panel(env, frame, rows, cols, protocol)
    return True


def encode_route_resize_rotate_and_acquire_socket(env):
    """Route a base agent into a shaped socket, resize, rotate, and acquire."""
    frame = env.frame()

    _, rows, cols = find_right_segment_panel(frame)
    if len(rows) != 6 or not cols:
        return False

    board_tiles = find_upper_right_board_tiles(frame)
    sources = find_upper_right_board_objects(frame)
    sockets = find_upper_right_components(frame, colors=(15,))
    if not board_tiles or not sources or not sockets:
        return False

    step = min(blob.size[0] for blob in board_tiles)
    source = max(sources, key=lambda blob: blob.area)
    socket = max(sockets, key=lambda blob: blob.area)

    r0, c0, r1, c1 = socket.bbox
    open_cells = {
        (row, col)
        for row in range(r0, r1 + 1)
        for col in range(c0, c1 + 1)
        if int(frame[row][col]) != socket.color
    }
    frontier = [(r0, col) for col in range(c0, c1 + 1) if (r0, col) in open_cells]
    cavity = set(frontier)
    while frontier:
        row, col = frontier.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = (row + dr, col + dc)
            if neighbor in open_cells and neighbor not in cavity:
                cavity.add(neighbor)
                frontier.append(neighbor)
    if not cavity:
        return False

    cavity_rows = [row for row, _ in cavity]
    cavity_cols = [col for _, col in cavity]
    cavity_bbox = (
        min(cavity_rows),
        min(cavity_cols),
        max(cavity_rows),
        max(cavity_cols),
    )
    cavity_height = cavity_bbox[2] - cavity_bbox[0] + 1
    cavity_width = cavity_bbox[3] - cavity_bbox[1] + 1
    if cavity_height != cavity_width or cavity_height % step:
        return False

    desired_scale = cavity_height // step
    source_scale = source.size[0] // step
    if (
        source.size[0] != source.size[1]
        or source.size[0] % step
        or source_scale <= 0
    ):
        return False

    def normalized_mask(bbox, color, scale):
        top, left, bottom, right = bbox
        mask = []
        for row in range(top, bottom + 1, scale):
            mask.append(
                tuple(
                    all(
                        int(frame[r][c]) == color
                        for r in range(row, row + scale)
                        for c in range(col, col + scale)
                    )
                    for col in range(left, right + 1, scale)
                )
            )
        return tuple(mask)

    source_shape = normalized_mask(source.bbox, source.color, source_scale)
    target_shape = tuple(
        tuple(
            all(
                (r, c) in cavity
                for r in range(row, row + desired_scale)
                for c in range(col, col + desired_scale)
            )
            for col in range(cavity_bbox[1], cavity_bbox[3] + 1, desired_scale)
        )
        for row in range(cavity_bbox[0], cavity_bbox[2] + 1, desired_scale)
    )

    rotations = None
    rotated = source_shape
    for turns in range(4):
        if rotated == target_shape:
            rotations = turns
            break
        rotated = tuple(zip(*rotated[::-1]))
    if rotations is None:
        return False

    start = source.top_left
    destination = cavity_bbox[:2]
    route = find_lattice_route(
        frame,
        start,
        destination,
        step,
        len(cols),
        ("D", "R", "L", "U"),
    )
    if route is None:
        return False

    if desired_scale >= source_scale:
        resize = ["X"] * (desired_scale - source_scale)
    else:
        resize = ["M"] * (source_scale - desired_scale)
    protocol = route + resize + ["A"] * rotations + ["B"]
    if len(protocol) != len(cols):
        return False

    write_protocol_to_segment_panel(env, frame, rows, cols, protocol)
    return True


def learn_direction_protocol_from_selector(env):
    """Read direction glyphs from the lower-left selector examples."""
    frame = env.frame()
    height, width = len(frame), len(frame[0])
    segments = [
        blob
        for blob in connected_components(frame, colors=(1, 5), min_area=3)
        if blob.area == 3
        and 1 in blob.size
        and blob.centroid[1] < width / 2
        and blob.centroid[0] > height / 2
    ]
    rows = sorted({int(round(blob.centroid[0])) for blob in segments})
    cols = sorted({int(round(blob.centroid[1])) for blob in segments})
    if len(rows) != 6 or not cols:
        return None

    selected_boxes = [
        blob
        for blob in connected_components(frame, colors=(9,), min_area=4)
        if blob.size == (9, 9)
        and blob.area == 32
        and blob.centroid[0] > height * 0.75
        and blob.centroid[1] < width / 2
    ]
    if len(selected_boxes) != 1:
        return None
    selected = selected_boxes[0]
    selector_row = int(round(selected.centroid[0]))
    selector_col = int(round(selected.centroid[1]))
    spacing = selected.size[1] + 1

    icons = [
        blob
        for blob in connected_components(frame, colors=(11,), min_area=4)
        if blob.area <= 9
        and blob.centroid[0] > height * 0.8
        and blob.centroid[1] < width * 0.65
    ]
    selectors = {}
    for icon in icons:
        box_col = selector_col + int(
            round((icon.centroid[1] - selector_col) / spacing)
        ) * spacing
        dr = icon.centroid[0] - selector_row
        dc = icon.centroid[1] - box_col
        if abs(dr) > abs(dc):
            direction = "D" if dr > 0 else "U"
        elif abs(dc) > 0:
            direction = "R" if dc > 0 else "L"
        else:
            continue
        selectors[direction] = (box_col, selector_row)
    if set(selectors) != {"D", "U", "L", "R"}:
        return None

    protocol = {}
    for direction in ("D", "U", "R", "L"):
        col, row = selectors[direction]
        env.step(6, col, row)
        current = env.frame()
        patterns = {
            tuple(
                index
                for index, segment_row in enumerate(rows)
                if int(current[segment_row][segment_col]) == 5
            )
            for segment_col in cols
        }
        if len(patterns) != 1:
            return None
        protocol[direction] = patterns.pop()
    return protocol


def route_to_socket_via_reacquisition_checkpoints(env):
    """Route through bounded checker waypoints, reacquiring between programs."""
    direction_protocol = learn_direction_protocol_from_selector(env)
    if direction_protocol is None:
        return False

    frame = env.frame()
    _, rows, cols = find_right_segment_panel(frame)
    board_tiles = find_upper_right_board_tiles(frame)
    board_objects = find_upper_right_board_objects(frame)
    if len(rows) != 6 or not cols or not board_tiles or not board_objects:
        return False

    step = min(blob.size[0] for blob in board_tiles)
    sources = [
        blob
        for blob in board_objects
        if blob.size == (step, step) and blob.area > step * step / 2
    ]
    targets = [blob for blob in board_objects if blob not in sources]
    if not sources or not targets:
        return False

    source = max(sources, key=lambda blob: blob.area)
    target = max(targets, key=lambda blob: blob.area)
    start = source.top_left

    destination = (
        snap_to_lattice(target.centroid[0], start[0], step),
        snap_to_lattice(target.centroid[1], start[1], step),
    )

    checkpoints = []
    for row in range(
        start[0] % step,
        len(frame) // 2 - step + 1,
        step,
    ):
        for col in range(
            start[1] % step,
            len(frame[0]) - step + 1,
            step,
        ):
            patch = [
                int(frame[r][c])
                for r in range(row, row + step)
                for c in range(col, col + step)
            ]
            if (
                patch.count(11) == step * step // 2
                and set(patch).issubset({4, 5, 11})
            ):
                checkpoints.append((row, col))
    if not checkpoints:
        return False

    queue = deque([(start, [])])
    seen = {start}
    plan = None
    while queue:
        position, stages = queue.popleft()
        max_steps = len(cols) if not stages else len(cols) - 1
        candidates = [destination] + checkpoints
        for waypoint in candidates:
            if waypoint == position:
                continue
            horizontal = (
                ("L", "R")
                if waypoint[1] < position[1]
                else ("R", "L")
            )
            vertical = (
                ("U", "D")
                if waypoint[0] < position[0]
                else ("D", "U")
            )
            route = find_lattice_route(
                frame,
                position,
                waypoint,
                step,
                max_steps,
                horizontal + vertical,
            )
            if route is None or len(route) > max_steps:
                continue
            next_stages = stages + [(waypoint, route)]
            if waypoint == destination:
                plan = next_stages
                queue.clear()
                break
            if waypoint not in seen:
                seen.add(waypoint)
                queue.append((waypoint, next_stages))
        if plan is not None:
            break
    if plan is None:
        return False

    protocol_rows = dict(PROTOCOL_ROWS)
    protocol_rows.update(direction_protocol)
    base_level = env.levels_completed
    for waypoint, route in plan:
        protocol = ["M"] * (len(cols) - len(route)) + route
        current = env.frame()
        configure_then_submit(
            env,
            lambda active_env: write_protocol_to_segment_panel(
                active_env,
                current,
                rows,
                cols,
                protocol,
                protocol_rows,
            ),
        )
        if env.levels_completed > base_level:
            return True

        staged = [
            blob
            for blob in find_upper_right_board_objects(env.frame())
            if blob.top_left == waypoint
            and blob.size == (step, step)
            and blob.area > step * step / 2
        ]
        if not staged:
            return False
    return False
