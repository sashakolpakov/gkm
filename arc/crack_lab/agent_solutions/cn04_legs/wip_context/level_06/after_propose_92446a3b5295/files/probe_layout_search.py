from collections import Counter


PIECES = [
    (
        {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 4),
         (2, 0), (2, 4), (3, 0), (4, 0), (4, 1), (4, 2), (5, 0)},
        {(2, 4), (4, 2), (5, 0)},
    ),
    (
        {(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 3), (2, 0),
         (2, 3), (3, 2), (3, 3)},
        {(2, 0), (3, 2)},
    ),
    (
        {(0, 2), (1, 1), (1, 2), (2, 0), (2, 1), (3, 1), (4, 1)},
        {(0, 2), (2, 0)},
    ),
    (
        {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (5, 1)},
        {(0, 0), (5, 1)},
    ),
    (
        {(0, 2), (0, 4), (1, 2), (1, 4), (2, 1), (2, 2), (2, 4),
         (2, 5), (3, 0), (3, 1), (3, 5), (4, 1), (4, 2), (4, 3),
         (4, 4), (4, 5)},
        {(0, 2), (0, 4), (3, 0)},
    ),
]


def rotate_piece(occupied, green):
    turned = {(c, -r) for r, c in occupied}
    turned_green = {(c, -r) for r, c in green}
    r0 = min(r for r, _ in turned)
    c0 = min(c for _, c in turned)
    return (
        {(r - r0, c - c0) for r, c in turned},
        {(r - r0, c - c0) for r, c in turned_green},
    )


def orientations(piece):
    occupied, green = piece
    out = []
    for turns in range(4):
        state = (frozenset(occupied), frozenset(green))
        if state not in out:
            out.append(state)
        occupied, green = rotate_piece(occupied, green)
    return out


ORIENTATIONS = [orientations(piece) for piece in PIECES]


def shifted(points, dr, dc):
    return {(r + dr, c + dc) for r, c in points}


def search():
    occupied = set(PIECES[0][0])
    green_counts = Counter(PIECES[0][1])
    placements = {0: (0, 0, 0)}
    seen = set()

    def visit(remaining, occupied, green_counts, placements):
        key = (
            tuple(sorted(remaining)),
            tuple(sorted(occupied)),
            tuple(sorted(green_counts.items())),
        )
        if key in seen:
            return None
        seen.add(key)
        if not remaining:
            if all(amount >= 2 for amount in green_counts.values()):
                return placements
            return None

        unmatched = sum(amount == 1 for amount in green_counts.values())
        endpoints_left = sum(len(PIECES[index][1]) for index in remaining)
        if unmatched > endpoints_left:
            return None

        assembly_green = set(green_counts)
        for index in remaining:
            next_remaining = tuple(i for i in remaining if i != index)
            for turns, (shape, green) in enumerate(ORIENTATIONS[index]):
                for source in green:
                    for target in assembly_green:
                        dr, dc = target[0] - source[0], target[1] - source[1]
                        moved_shape = shifted(shape, dr, dc)
                        moved_green = shifted(green, dr, dc)
                        overlap = moved_green & assembly_green
                        if not overlap:
                            continue
                        next_counts = green_counts.copy()
                        next_counts.update(moved_green)
                        found = visit(
                            next_remaining,
                            occupied | moved_shape,
                            next_counts,
                            {**placements, index: (turns, dr, dc)},
                        )
                        if found is not None:
                            return found
        return None

    result = visit(tuple(range(1, len(PIECES))), occupied, green_counts, placements)
    print("states", len(seen), "result", result)
    if result is not None:
        for index, (turns, dr, dc) in sorted(result.items()):
            green = ORIENTATIONS[index][turns][1]
            print(index, sorted(shifted(green, dr, dc)))


search()
