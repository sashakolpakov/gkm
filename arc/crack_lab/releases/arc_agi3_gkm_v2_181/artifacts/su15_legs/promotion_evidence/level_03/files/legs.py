# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.
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
