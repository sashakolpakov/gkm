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
