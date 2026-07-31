# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.
import heapq
import itertools

from perception import connected_components


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

    def playfield_squares(color):
        return [
            blob for blob in connected_components(
                env.frame(), colors=(color,), min_area=1
            )
            if blob.bbox[0] >= 10
            and blob.size[0] == blob.size[1]
            and blob.area == blob.size[0] ** 2
        ]

    def anchor(blob):
        r0, c0, r1, c1 = blob.bbox
        return (r0 + r1 + 1) // 2, (c0 + c1 + 1) // 2

    def click(row, col):
        env.step(6, int(col), int(row))

    moves = 0
    for color in (10, 6, 15):
        while moves < max_moves:
            if env.terminal() or env.levels_completed != start_level:
                return
            squares = playfield_squares(color)
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
            click(row, col)
            moves += 1

    while moves < max_moves:
        if env.terminal() or env.levels_completed != start_level:
            return
        final = playfield_squares(11)
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
        click(row, col)
        moves += 1


def deliver_remaining_square_via_diagonal_detour(env, step=6, max_moves=16):
    """Route a remaining diagonal-moving square around occupied target rings."""
    start_level = env.levels_completed

    def pieces():
        return [
            blob for blob in connected_components(
                env.frame(), colors=(15,), min_area=9
            )
            if blob.bbox[0] >= 10
            and blob.size[0] == blob.size[1]
            and blob.area == blob.size[0] ** 2
        ]

    def rings():
        return [
            blob for blob in connected_components(
                env.frame(), colors=(9,), min_area=9
            )
            if blob.bbox[0] >= 10
        ]

    def click(row, col):
        env.step(6, int(round(col)), int(round(row)))

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
        click(row + vertical, col + horizontal)
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
        click(row + step, next_col)
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
        return [
            blob for blob in connected_components(env.frame(), min_area=1)
            if (
                blob.bbox[0] >= 10
                and blob.size[0] == blob.size[1]
                and blob.area == blob.size[0] ** 2
                and blob.color not in (3, 4, 5, 7, 9)
            )
        ]

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
        next_row = row + max(-step, min(step, target_row - row))
        next_col = col + max(-step, min(step, target_col - col))
        env.step(6, int(round(next_col)), int(round(next_row)))
        moves += 1

    while moves < max_moves:
        if env.terminal() or env.levels_completed != start_level:
            return
        row, col = alternate.centroid
        env.step(6, int(round(col)), int(round(row)))
        moves += 1
