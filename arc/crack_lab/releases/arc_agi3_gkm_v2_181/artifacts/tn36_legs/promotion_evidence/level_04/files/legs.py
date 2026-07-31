# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.

from collections import deque

from perception import connected_components


def make_small_segments_color_5_and_submit(env):
    """Toggle each small color-1 line segment, then click the color-9 disc."""
    segments = [
        blob
        for blob in connected_components(env.frame(), colors=(1,), min_area=3)
        if blob.area == 3 and 1 in blob.size
    ]
    for blob in segments:
        row, col = blob.centroid
        env.step(6, int(round(col)), int(round(row)))

    submit_discs = [
        blob
        for blob in connected_components(env.frame(), colors=(9,), min_area=4)
        if blob.size[0] > 1 and blob.size[1] > 1
    ]
    if submit_discs:
        disc = max(submit_discs, key=lambda blob: blob.area)
        row, col = disc.centroid
        env.step(6, int(round(col)), int(round(row)))


def turn_on_outer_rows_of_right_segment_panel_and_submit(env):
    """Turn on the top and bottom rows of the right segment panel, then submit."""
    frame = env.frame()
    width = len(frame[0])
    segments = [
        blob
        for blob in connected_components(frame, colors=(1, 5), min_area=3)
        if blob.area == 3
        and 1 in blob.size
        and blob.centroid[1] > width / 2
    ]
    if not segments:
        return

    rows = [blob.centroid[0] for blob in segments]
    outer_rows = (min(rows), max(rows))
    for blob in segments:
        if blob.color == 1 and any(
            abs(blob.centroid[0] - row) < 0.5 for row in outer_rows
        ):
            row, col = blob.centroid
            env.step(6, int(round(col)), int(round(row)))

    submit_discs = [
        blob
        for blob in connected_components(frame, colors=(9,), min_area=4)
        if blob.size[0] > 1 and blob.size[1] > 1
    ]
    if submit_discs:
        disc = max(submit_discs, key=lambda blob: blob.area)
        row, col = disc.centroid
        env.step(6, int(round(col)), int(round(row)))


def encode_reacquisition_route_through_barrier_and_submit(env):
    """Route the small cyan agent through a wall gap and encode the route."""
    frame = env.frame()
    height, width = len(frame), len(frame[0])

    segments = [
        blob
        for blob in connected_components(frame, colors=(1, 5), min_area=3)
        if blob.area == 3 and 1 in blob.size and blob.centroid[1] > width / 2
    ]
    rows = sorted({int(round(blob.centroid[0])) for blob in segments})
    cols = sorted({int(round(blob.centroid[1])) for blob in segments})

    board_objects = [
        blob
        for blob in connected_components(frame, colors=(11,), min_area=4)
        if blob.bbox[2] < height / 2 and blob.bbox[1] > width / 2
    ]
    sources = [
        blob for blob in board_objects if blob.area == 14 and blob.size == (4, 4)
    ]
    targets = [blob for blob in board_objects if blob not in sources]
    if len(rows) != 6 or not cols or not sources or not targets:
        return

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
            return
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
        return

    direction_rows = {
        "D": (0, 1),
        "U": (0, 5),
        "L": (1, 5),
        "R": (1,),
    }
    for col_index, direction in enumerate(route):
        desired_rows = direction_rows[direction]
        for row_index, row in enumerate(rows):
            is_on = int(frame[row][cols[col_index]]) == 5
            should_be_on = row_index in desired_rows
            if is_on != should_be_on:
                env.step(6, cols[col_index], row)

    submit_discs = [
        blob
        for blob in connected_components(env.frame(), colors=(9,), min_area=4)
        if blob.size[0] > 1 and blob.size[1] > 1
    ]
    if submit_discs:
        disc = max(submit_discs, key=lambda blob: blob.area)
        row, col = disc.centroid
        env.step(6, int(round(col)), int(round(row)))


def reacquire_max_scaled_agent_and_route_to_socket_then_submit(env):
    """Collapse a scaled agent, route one lattice cell at a time, and submit."""
    frame = env.frame()
    height, width = len(frame), len(frame[0])

    segments = [
        blob
        for blob in connected_components(frame, colors=(1, 5), min_area=3)
        if blob.area == 3 and 1 in blob.size and blob.centroid[1] > width / 2
    ]
    rows = sorted({int(round(blob.centroid[0])) for blob in segments})
    cols = sorted({int(round(blob.centroid[1])) for blob in segments})
    if len(rows) != 6 or not cols:
        return

    board_tiles = [
        blob
        for blob in connected_components(frame, colors=(4,), min_area=9)
        if blob.bbox[2] < height / 2
        and blob.bbox[1] > width / 2
        and blob.size[0] == blob.size[1]
        and blob.area == blob.size[0] * blob.size[1]
    ]
    if not board_tiles:
        return
    step = min(blob.size[0] for blob in board_tiles)

    board_objects = [
        blob
        for blob in connected_components(frame, colors=(11,), min_area=4)
        if blob.bbox[2] < height / 2 and blob.bbox[1] > width / 2
    ]
    sources = [
        blob
        for blob in board_objects
        if blob.size[0] == blob.size[1] and blob.size[0] >= step
    ]
    targets = [blob for blob in board_objects if blob not in sources]
    if not sources or not targets:
        return

    source = max(sources, key=lambda blob: blob.area)
    target = max(targets, key=lambda blob: blob.area)
    scale = source.size[0] // step
    reacquisitions = max(0, scale - 1)

    start = source.top_left
    destination = (target.bbox[0], target.bbox[1] + 1)
    direction_delta = {
        "L": (0, -step),
        "R": (0, step),
        "U": (-step, 0),
        "D": (step, 0),
    }

    horizontal = ("L", "R") if destination[1] < start[1] else ("R", "L")
    vertical = ("U", "D") if destination[0] < start[0] else ("D", "U")
    direction_order = horizontal + vertical

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
    route = None
    while queue:
        position, path = queue.popleft()
        if position == destination:
            route = path
            break
        if len(path) >= len(cols) - reacquisitions:
            continue
        for direction in direction_order:
            dr, dc = direction_delta[direction]
            neighbor = (position[0] + dr, position[1] + dc)
            if neighbor in seen or not walkable(neighbor):
                continue
            seen.add(neighbor)
            queue.append((neighbor, path + [direction]))

    protocol = ["M"] * reacquisitions + (route or [])
    if route is None or len(protocol) != len(cols):
        return

    direction_rows = {
        "D": (0, 1),
        "U": (0, 5),
        "L": (1, 5),
        "R": (1,),
        "M": (0, 3),
    }
    for col_index, direction in enumerate(protocol):
        desired_rows = direction_rows[direction]
        for row_index, row in enumerate(rows):
            is_on = int(frame[row][cols[col_index]]) == 5
            should_be_on = row_index in desired_rows
            if is_on != should_be_on:
                env.step(6, cols[col_index], row)

    submit_discs = [
        blob
        for blob in connected_components(env.frame(), colors=(9,), min_area=4)
        if blob.size[0] > 1 and blob.size[1] > 1
    ]
    if submit_discs:
        disc = max(submit_discs, key=lambda blob: blob.area)
        row, col = disc.centroid
        env.step(6, int(round(col)), int(round(row)))
