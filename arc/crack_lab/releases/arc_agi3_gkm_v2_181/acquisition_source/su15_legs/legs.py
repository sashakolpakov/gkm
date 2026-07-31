# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.
import heapq
import itertools

from perception import connected_components


def _solid_playfield_squares(
    env, colors=None, excluded_colors=(), min_area=1
):
    return [
        blob for blob in connected_components(
            env.frame(), colors=colors, min_area=min_area
        )
        if (
            blob.bbox[0] >= 10
            and blob.size[0] == blob.size[1]
            and blob.area == blob.size[0] ** 2
            and blob.color not in excluded_colors
        )
    ]


def _click(env, row, col):
    env.step(6, int(round(col)), int(round(row)))


def _move_square_one_step(env, square, target, step=None):
    row, col = square.centroid
    target_row, target_col = target
    if step is None:
        step = square.size[0] + 1
    _click(
        env,
        row + max(-step, min(step, target_row - row)),
        col + max(-step, min(step, target_col - col)),
    )


def _body_groups(env, color):
    frame = env.frame()
    remaining = {
        (row, col)
        for row in range(10, len(frame))
        for col in range(len(frame[0]))
        if int(frame[row][col]) == color
    }
    found = []
    while remaining:
        todo = [remaining.pop()]
        group = []
        while todo:
            row, col = todo.pop()
            group.append((row, col))
            near = {
                point for point in remaining
                if max(abs(point[0] - row), abs(point[1] - col)) <= 1
            }
            remaining -= near
            todo.extend(near)
        if len(group) >= 4:
            found.append(tuple(sorted(group)))
    return tuple(sorted(found))


def _body_center(points):
    return (
        round(sum(row for row, _ in points) / len(points)),
        round(sum(col for _, col in points) / len(points)),
    )


def _level_active(env, start_level):
    return not env.terminal() and env.levels_completed == start_level


def _control_body(env, start_level, color, offset, selector=None):
    if not _level_active(env, start_level):
        return False
    groups = _body_groups(env, color)
    if not groups:
        return False
    group = selector(groups) if selector is not None else groups[0]
    row, col = _body_center(group)
    point = (row + offset[0], col + offset[1])
    if point not in group:
        return False
    _click(env, *point)
    return True


def _control_body_sequence(env, start_level, controls):
    """Apply an ordered sequence of moving-body redirects."""
    for control in controls:
        color, offset, *selector = control
        if not _control_body(
            env,
            start_level,
            color,
            offset,
            selector[0] if selector else None,
        ):
            return False
    return True


def _solid_square_is_at_target(env, color, target):
    """Return whether the sole solid square of a color is seated at a target."""
    squares = _solid_playfield_squares(env, colors=(color,))
    return (
        len(squares) == 1
        and max(
            abs(round(squares[0].centroid[0]) - target[0]),
            abs(round(squares[0].centroid[1]) - target[1]),
        ) <= 1
    )


def follow_diagonal_lattice_to_ring(env, step=6, max_moves=12):
    """Move the playfield avatar diagonally toward the ringed target."""
    start_level = env.levels_completed
    for _ in range(max_moves):
        if env.terminal() or env.levels_completed != start_level:
            return

        frame = env.frame()
        avatars = [
            blob for blob in connected_components(frame, colors=(15,), min_area=9)
            if blob.area == 9 and blob.centroid[0] >= 10
        ]
        targets = connected_components(frame, colors=(3,), min_area=9)
        if not avatars or not targets:
            return

        avatar = max(avatars, key=lambda blob: blob.centroid[0])
        target = min(
            targets,
            key=lambda blob: (
                (blob.centroid[0] - avatar.centroid[0]) ** 2
                + (blob.centroid[1] - avatar.centroid[1]) ** 2
            ),
        )
        ar, ac = avatar.centroid
        tr, tc = target.centroid
        dr = 0 if tr == ar else (step if tr > ar else -step)
        dc = 0 if tc == ac else (step if tc > ac else -step)
        env.step(6, int(round(ac + dc)), int(round(ar + dr)))


def merge_equal_squares_and_deliver_to_ring(env, step=6, max_moves=32):
    """Merge equal playfield squares, then carry the result into its ring."""
    start_level = env.levels_completed

    def anchor(blob):
        r0, c0, r1, c1 = blob.bbox
        return (r0 + r1 + 1) // 2, (c0 + c1 + 1) // 2

    moves = 0
    for color in (10, 6, 15):
        while moves < max_moves:
            if env.terminal() or env.levels_completed != start_level:
                return
            squares = _solid_playfield_squares(env, colors=(color,))
            if len(squares) < 2:
                break
            first, second = min(
                (
                    (left, right)
                    for index, left in enumerate(squares)
                    for right in squares[index + 1:]
                ),
                key=lambda pair: max(
                    abs(anchor(pair[0])[0] - anchor(pair[1])[0]),
                    abs(anchor(pair[0])[1] - anchor(pair[1])[1]),
                ),
            )
            fr, fc = anchor(first)
            sr, sc = anchor(second)
            distance = max(abs(fr - sr), abs(fc - sc))
            if distance <= 2 * step:
                row = round((fr + sr) / 2)
                col = round((fc + sc) / 2)
            else:
                row = sr + max(-step, min(step, fr - sr))
                col = sc + max(-step, min(step, fc - sc))
            _click(env, row, col)
            moves += 1

    while moves < max_moves:
        if env.terminal() or env.levels_completed != start_level:
            return
        final = _solid_playfield_squares(env, colors=(11,))
        rings = connected_components(env.frame(), colors=(9,), min_area=9)
        if not final or not rings:
            return
        square = max(final, key=lambda blob: blob.area)
        ring = min(
            rings,
            key=lambda blob: (
                (blob.centroid[0] - square.centroid[0]) ** 2
                + (blob.centroid[1] - square.centroid[1]) ** 2
            ),
        )
        sr, sc = anchor(square)
        rr, rc = ring.centroid
        row = sr + max(-step, min(step, round(rr) - sr))
        col = sc + max(-step, min(step, round(rc) - sc))
        _click(env, row, col)
        moves += 1


def deliver_remaining_square_via_diagonal_detour(env, step=6, max_moves=16):
    """Route a remaining diagonal-moving square around occupied target rings."""
    start_level = env.levels_completed

    def pieces():
        return _solid_playfield_squares(env, colors=(15,), min_area=9)

    def rings():
        return [
            blob for blob in connected_components(
                env.frame(), colors=(9,), min_area=9
            )
            if blob.bbox[0] >= 10
        ]

    movable = pieces()
    targets = rings()
    if not movable or not targets:
        return

    piece = max(movable, key=lambda blob: blob.area)
    target = max(targets, key=lambda blob: blob.area)
    _, start_col = piece.centroid
    _, target_col = target.centroid
    center_lane = start_col + round((target_col - start_col) / step) * step

    other_rings = [ring for ring in targets if ring != target]
    if other_rings:
        other_col = sum(ring.centroid[1] for ring in other_rings) / len(other_rings)
        outward = -1 if other_col > target_col else 1
    else:
        outward = -1 if start_col > target_col else 1
    outside_lane = center_lane + outward * step
    high_turn_row = target.bbox[0] - 2 * step

    moves = 0
    while moves < max_moves:
        if env.terminal() or env.levels_completed != start_level:
            return
        movable = pieces()
        if not movable:
            return
        piece = max(movable, key=lambda blob: blob.area)
        row, col = piece.centroid
        if abs(col - center_lane) <= 1:
            break
        horizontal = step if center_lane > col else -step
        vertical = step if row <= high_turn_row else -step
        _click(env, row + vertical, col + horizontal)
        moves += 1

    while moves < max_moves:
        if env.terminal() or env.levels_completed != start_level:
            return
        movable = pieces()
        if not movable:
            return
        piece = max(movable, key=lambda blob: blob.area)
        row, col = piece.centroid
        next_col = outside_lane if abs(col - center_lane) <= 1 else center_lane
        _click(env, row + step, next_col)
        moves += 1


def merge_equal_squares_around_moving_cutter(
    env, step=6, max_states=5000, max_depth=30, minimum_stage_mass=1
):
    """Plan collision-free equal-square merges, then deliver into the ring.

    ``minimum_stage_mass`` selects one merge cohort when smaller, unrelated
    squares share the playfield.
    """
    start_level = env.levels_completed
    stages = {
        color: mass
        for color, mass in ((10, 1), (6, 2), (15, 4), (11, 8))
        if mass >= minimum_stage_mass
    }

    def pieces(node):
        found = []
        for color, mass in stages.items():
            for blob in connected_components(
                node.frame(), colors=(color,), min_area=1
            ):
                if (
                    blob.bbox[0] >= 10
                    and blob.size[0] == blob.size[1]
                    and blob.area == blob.size[0] ** 2
                ):
                    found.append((color, mass, blob))
        return found

    def anchor(blob):
        r0, c0, r1, c1 = blob.bbox
        return (r0 + r1 + 1) // 2, (c0 + c1 + 1) // 2

    def avatar_centers(node):
        frame = node.frame()
        points = {
            (row, col)
            for row in range(10, len(frame))
            for col in range(len(frame[0]))
            if int(frame[row][col]) == 7
        }
        groups = []
        while points:
            todo = [points.pop()]
            group = []
            while todo:
                point = todo.pop()
                group.append(point)
                near = {
                    other for other in points
                    if max(
                        abs(point[0] - other[0]),
                        abs(point[1] - other[1]),
                    ) <= 1
                }
                points -= near
                todo.extend(near)
            groups.append(
                (
                    round(sum(row for row, _ in group) / len(group)),
                    round(sum(col for _, col in group) / len(group)),
                )
            )
        return tuple(sorted(groups))

    def rings(node):
        return [
            blob for blob in connected_components(
                node.frame(), colors=(9,), min_area=9
            )
            if blob.bbox[0] >= 10
        ]

    def state(node):
        movable = pieces(node)
        compact = tuple(
            sorted((color, blob.bbox) for color, _, blob in movable)
        )
        return movable, avatar_centers(node), compact

    def target_for(node, blob):
        targets = rings(node)
        if not targets:
            return None
        row, col = anchor(blob)
        return min(
            targets,
            key=lambda target: max(
                abs(row - round(target.centroid[0])),
                abs(col - round(target.centroid[1])),
            ),
        )

    def candidate_actions(node):
        movable, avatars, _ = state(node)
        proposed = []
        for color in stages:
            same = [
                blob for piece_color, _, blob in movable
                if piece_color == color
            ]
            pairs = []
            for index, left in enumerate(same):
                for right in same[index + 1:]:
                    lr, lc = anchor(left)
                    rr, rc = anchor(right)
                    distance = max(abs(lr - rr), abs(lc - rc))
                    pairs.append((distance, left, right))
            for distance, left, right in sorted(
                pairs, key=lambda item: item[0]
            )[:6]:
                for first, second in ((left, right), (right, left)):
                    fr, fc = anchor(first)
                    sr, sc = anchor(second)
                    if distance <= 2 * step:
                        row = round((fr + sr) / 2)
                        col = round((fc + sc) / 2)
                    else:
                        row = sr + max(-step, min(step, fr - sr))
                        col = sc + max(-step, min(step, fc - sc))
                    proposed.append((6, int(col), int(row)))

        for color, _, final in movable:
            if color != 11:
                continue
            target = target_for(node, final)
            if target is None:
                continue
            row, col = anchor(final)
            target_row = round(target.centroid[0])
            target_col = round(target.centroid[1])
            proposed.append(
                (
                    6,
                    int(col + max(-step, min(step, target_col - col))),
                    int(row + max(-step, min(step, target_row - row))),
                )
            )

        frame = node.frame()
        rows, cols = len(frame), len(frame[0])
        top, bottom = 13, rows - 5
        left, right = 4, cols - 5
        lower_left = left
        if any(
            max(
                abs(bottom - round(ring.centroid[0])),
                abs(left - round(ring.centroid[1])),
            ) <= 2 * step
            for ring in rings(node)
        ):
            lower_left += 2 * step
        proposed.extend(
            (6, col, row)
            for row, col in (
                (top, left),
                (top, right),
                (bottom, right),
                (bottom, lower_left),
            )
        )
        proposed.extend((6, col, row) for row, col in avatars)
        return list(dict.fromkeys(proposed))

    def priority(node, depth):
        movable, avatar, _ = state(node)
        if not movable:
            return 1_000_000
        merge_distance = 0
        for color in stages:
            same = [
                blob for piece_color, _, blob in movable
                if piece_color == color
            ]
            if len(same) >= 2:
                distances = []
                for index, left in enumerate(same):
                    for right in same[index + 1:]:
                        lr, lc = anchor(left)
                        rr, rc = anchor(right)
                        distances.append(max(abs(lr - rr), abs(lc - rc)))
                merge_distance += min(distances)

        clearance = min(
            (
                max(
                    abs(avatar_row - anchor(blob)[0]),
                    abs(avatar_col - anchor(blob)[1]),
                )
                for avatar_row, avatar_col in avatar
                for _, _, blob in movable
            ),
            default=0,
        )
        ring_distance = 0
        if len(movable) == 1 and movable[0][0] == 11:
            final = movable[0][2]
            target = target_for(node, final)
            if target is not None:
                row, col = anchor(final)
                ring_distance = max(
                    abs(row - round(target.centroid[0])),
                    abs(col - round(target.centroid[1])),
                )
        return (
            len(movable) * 100
            + merge_distance
            + ring_distance
            - min(clearance, 30)
            + depth
        )

    start = env.clone()
    initial, avatar, compact = state(start)
    initial_mass = sum(mass for _, mass, _ in initial)
    serial = itertools.count()
    heap = [(priority(start, 0), 0, next(serial), start, [])]
    seen = {(compact, avatar)}
    path = None
    expanded = 0
    while heap and expanded < max_states:
        _, depth, _, node, prefix = heapq.heappop(heap)
        expanded += 1
        if depth >= max_depth:
            continue
        for action in candidate_actions(node):
            child = node.clone()
            try:
                child.step(*action)
            except Exception:
                continue
            child_path = prefix + [action]
            if child.levels_completed > start_level:
                path = child_path
                heap = []
                break
            movable, child_avatar, child_compact = state(child)
            if (
                child.terminal()
                or sum(mass for _, mass, _ in movable) != initial_mass
            ):
                continue
            key = (child_compact, child_avatar)
            if key in seen:
                continue
            seen.add(key)
            heapq.heappush(
                heap,
                (
                    priority(child, len(child_path)),
                    len(child_path),
                    next(serial),
                    child,
                    child_path,
                ),
            )

    if path is None:
        return
    for action in path:
        if env.terminal() or env.levels_completed != start_level:
            return
        env.step(*action)


def stage_large_square_for_diagonal_partner(env, max_moves=16):
    """Stage the largest square in the far ring, then advance its partner."""
    start_level = env.levels_completed

    def solid_squares():
        return _solid_playfield_squares(
            env, excluded_colors=(3, 4, 5, 7, 9)
        )

    rings = [
        blob for blob in connected_components(
            env.frame(), colors=(9,), min_area=9
        )
        if blob.bbox[0] >= 10
    ]
    squares = solid_squares()
    if len(rings) < 2 or not squares:
        return

    square = max(squares, key=lambda blob: blob.area)
    start_row, start_col = square.centroid
    target = max(
        rings,
        key=lambda ring: max(
            abs(ring.centroid[0] - start_row),
            abs(ring.centroid[1] - start_col),
        ),
    )
    alternate = max(
        (ring for ring in rings if ring != target),
        key=lambda ring: max(
            abs(ring.centroid[0] - target.centroid[0]),
            abs(ring.centroid[1] - target.centroid[1]),
        ),
    )
    step = square.size[0] + 1
    moves = 0

    while moves < max_moves:
        if env.terminal() or env.levels_completed != start_level:
            return
        squares = solid_squares()
        if not squares:
            return
        square = max(squares, key=lambda blob: blob.area)
        row, col = square.centroid
        target_row, target_col = target.centroid
        if max(abs(row - target_row), abs(col - target_col)) <= 1:
            break
        _move_square_one_step(
            env, square, (target_row, target_col), step=step
        )
        moves += 1

    while moves < max_moves:
        if env.terminal() or env.levels_completed != start_level:
            return
        row, col = alternate.centroid
        _click(env, row, col)
        moves += 1


def left_corner_ring_targets(
    env, pair_color, moving_body_count, minimum_ring_count
):
    """Identify the two left targets for a validated corner-ring layout."""
    rings = [
        blob for blob in connected_components(
            env.frame(), colors=(9,), min_area=9
        )
        if blob.bbox[0] >= 10
    ]
    pair = _solid_playfield_squares(env, colors=(pair_color,))
    large = _solid_playfield_squares(env, colors=(8,))
    if (
        len(rings) < minimum_ring_count
        or len(pair) != 2
        or len(large) != 1
        or len(_body_groups(env, 7)) != moving_body_count
    ):
        return None

    left_rings = sorted(rings, key=lambda ring: ring.centroid[1])[:2]
    upper_left = min(left_rings, key=lambda ring: ring.centroid[0])
    lower_left = max(left_rings, key=lambda ring: ring.centroid[0])
    return (
        tuple(map(round, upper_left.centroid)),
        tuple(map(round, lower_left.centroid)),
    )


def merge_small_squares_along_corner_lane(env, target):
    """Merge the nearby small pair along the lane to a corner target."""
    small = _solid_playfield_squares(env, colors=(11,))
    if len(small) != 2:
        return False
    _, target_col = target
    small_size = max(blob.size[0] for blob in small)
    _click(
        env,
        max(round(blob.centroid[0]) for blob in small) + small_size,
        target_col,
    )
    return len(_solid_playfield_squares(env, colors=(12,))) == 1


def move_solid_square_to_target(env, color, target, max_moves):
    """Move one solid square toward a target in size-derived strides."""
    start_level = env.levels_completed
    for _ in range(max_moves):
        if not _level_active(env, start_level):
            return False
        candidates = _solid_playfield_squares(env, colors=(color,))
        if not candidates:
            return False
        square = candidates[0]
        row, col = map(round, square.centroid)
        if max(abs(row - target[0]), abs(col - target[1])) <= 1:
            break
        _move_square_one_step(env, square, target)
    return True


def merge_moving_bodies_preserving_cutter(env):
    """Merge two moving bodies while retaining the third as a cutter."""
    start_level = env.levels_completed
    controls = (
        (
            7,
            (-1, 1),
            lambda groups: max(
                groups, key=lambda group: _body_center(group)[0]
            ),
        ),
        (
            7,
            (0, 2),
            lambda groups: max(
                groups, key=lambda group: _body_center(group)[1]
            ),
        ),
        (
            7,
            (-1, 1),
            lambda groups: max(
                groups, key=lambda group: _body_center(group)[1]
            ),
        ),
    )
    if not _control_body_sequence(env, start_level, controls):
        return False
    return len(_body_groups(env, 7)) == 1 and len(_body_groups(env, 14)) == 1


def reseat_square_while_cutting_staged_square(env, target):
    """Reseat one square while the remaining body cuts the staged square."""
    start_level = env.levels_completed
    merged = _solid_playfield_squares(env, colors=(12,))
    if len(merged) != 1:
        return False
    square = merged[0]
    row, col = map(round, square.centroid)
    step = square.size[0] + 1
    _click(env, row, col + max(-step, min(step, target[1] - col)))
    if not _level_active(env, start_level):
        return False
    _click(env, *target)

    return (
        len(_solid_playfield_squares(env, colors=(12,))) == 2
        and len(_body_groups(env, 7)) == 1
        and len(_body_groups(env, 14)) == 1
    )


def route_cutter_and_merged_body_to_corner_rings(env):
    """Redirect the coupled bodies while keeping both squares seated."""
    start_level = env.levels_completed
    controls = (
        (14, (0, 2)),
        (7, (1, 0)),
        (7, (1, 0)),
        (7, (0, 2)),
        (7, (1, 0)),
        (7, (1, 1)),
        (7, (0, 2)),
        (7, (-1, -1)),
        (7, (1, 0)),
        (7, (-1, 1)),
        (14, (0, 0)),
        (7, (0, 2)),
        (7, (-2, 0)),
    )
    return _control_body_sequence(env, start_level, controls)


def merge_square_pair_at_anchor(env, color):
    """Merge one equal-square pair by selecting an anchor square."""
    start_level = env.levels_completed
    squares = _solid_playfield_squares(env, colors=(color,))
    if len(squares) != 2:
        return False
    row, col = squares[0].bbox[:2]
    _click(env, row, col)
    return (
        _level_active(env, start_level)
        and len(_solid_playfield_squares(env, colors=(color,))) == 0
    )


def route_large_square_via_northwest_lane(env, color, target):
    """Route a large square around a divider via its northwest lattice lane."""
    start_level = env.levels_completed
    squares = _solid_playfield_squares(env, colors=(color,))
    if len(squares) != 1:
        return False
    step = squares[0].size[0] + 1

    for row_step, col_step in (
        (-step, -step),
        (-step, -step),
        (0, -step),
    ):
        if not _level_active(env, start_level):
            return False
        squares = _solid_playfield_squares(env, colors=(color,))
        if len(squares) != 1:
            return False
        row, col = map(round, squares[0].centroid)
        _click(env, row + row_step, col + col_step)

    if not _level_active(env, start_level):
        return False
    squares = _solid_playfield_squares(env, colors=(color,))
    if len(squares) != 1:
        return False
    _move_square_one_step(env, squares[0], target)
    return _solid_square_is_at_target(env, color, target)


def merge_four_moving_bodies_beside_staged_square(env):
    """Reduce four moving bodies while allowing a staged square to be nudged."""
    start_level = env.levels_completed
    controls = (
        (
            7,
            (0, -2),
            lambda groups: max(
                groups, key=lambda group: _body_center(group)[0]
            ),
        ),
        (
            7,
            (-1, 1),
            lambda groups: min(
                groups, key=lambda group: _body_center(group)[1]
            ),
        ),
        (
            14,
            (-2, 0),
            lambda groups: max(
                groups, key=lambda group: _body_center(group)[1]
            ),
        ),
    )
    if not _control_body_sequence(env, start_level, controls):
        return False
    return (
        not _body_groups(env, 7)
        and not _body_groups(env, 14)
        and len(_body_groups(env, 13)) == 1
    )


def clear_final_body_and_reseat_large_square(env, color, target):
    """Move a merged body clear, then return a displaced square to its ring."""
    start_level = env.levels_completed
    for _ in range(3):
        if not _control_body(env, start_level, 13, (-2, 2)):
            return False

    squares = _solid_playfield_squares(env, colors=(color,))
    if len(squares) != 1:
        return False
    _move_square_one_step(env, squares[0], target)
    return (
        _level_active(env, start_level)
        and len(_body_groups(env, 13)) == 1
        and _solid_square_is_at_target(env, color, target)
    )


def advance_autonomous_bodies(env, turns=1):
    """Advance uncontrolled moving bodies with neutral playfield clicks."""
    start_level = env.levels_completed
    for _ in range(turns):
        if not _level_active(env, start_level):
            break
        _click(env, 32, 32)
    return env.levels_completed > start_level
