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


def write_protocol_to_segment_panel(env, frame, rows, cols, protocol):
    """Set panel columns to the six-row glyph for each protocol symbol."""
    for col_index, direction in enumerate(protocol):
        desired_rows = PROTOCOL_ROWS[direction]
        for row_index, row in enumerate(rows):
            is_on = int(frame[row][cols[col_index]]) == 5
            should_be_on = row_index in desired_rows
            if is_on != should_be_on:
                env.step(6, cols[col_index], row)


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


def configure_then_submit(env, configure):
    """Run a configuration leg and submit only when it succeeds."""
    if configure(env):
        click_largest_color_9_submit_disc(env)


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

    def snap_to_lattice(value, origin):
        return origin + int(round((value - origin) / step)) * step

    target_row = snap_to_lattice(target.centroid[0], source_row)
    target_col = snap_to_lattice(target.centroid[1], source_col)

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
        barrier_col = snap_to_lattice(barrier.centroid[1], source_col)
        if min(source_col, target_col) < barrier_col < max(source_col, target_col):
            separating.append((barrier_col, barrier))

    if separating:
        wall_col = separating[0][0]
        wall_top = min(blob.bbox[0] for _, blob in separating)
        wall_bottom = max(blob.bbox[2] for _, blob in separating)
        first_lattice_row = snap_to_lattice(wall_top, source_row)
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
