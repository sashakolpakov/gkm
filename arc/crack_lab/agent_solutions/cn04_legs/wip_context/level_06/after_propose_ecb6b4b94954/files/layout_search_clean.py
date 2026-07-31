"""Search exact green-pair registrations with no accidental body overlap."""
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


def rotate(shape, green):
    shape = {(col, -row) for row, col in shape}
    green = {(col, -row) for row, col in green}
    r0 = min(row for row, _ in shape)
    c0 = min(col for _, col in shape)
    return (
        {(row - r0, col - c0) for row, col in shape},
        {(row - r0, col - c0) for row, col in green},
    )


ORIENTATIONS = []
for piece in PIECES:
    choices = []
    for _ in range(4):
        choices.append(piece)
        piece = rotate(*piece)
    ORIENTATIONS.append(choices)


def shift(points, row, col):
    return {(r + row, c + col) for r, c in points}


def search(limit=30):
    shape, green = ORIENTATIONS[0][0]
    occupancy = {point: (point in green) for point in shape}
    green_counts = Counter(green)
    solutions = []

    def visit(remaining, occupancy, green_counts, placements):
        if len(solutions) >= limit:
            return
        if not remaining:
            if set(green_counts.values()) == {2}:
                solutions.append(dict(placements))
            return
        open_green = {point for point, count in green_counts.items() if count == 1}
        for index in remaining:
            rest = tuple(i for i in remaining if i != index)
            for turns, (local_shape, local_green) in enumerate(ORIENTATIONS[index]):
                candidates = set()
                for source in local_green:
                    for target in open_green:
                        candidates.add((target[0] - source[0],
                                        target[1] - source[1]))
                for dr, dc in candidates:
                    moved_shape = shift(local_shape, dr, dc)
                    moved_green = shift(local_green, dr, dc)
                    overlaps = moved_shape & set(occupancy)
                    if not overlaps:
                        continue
                    if not overlaps <= moved_green:
                        continue
                    if not all(occupancy[point] for point in overlaps):
                        continue
                    next_counts = green_counts.copy()
                    next_counts.update(moved_green)
                    if max(next_counts.values()) > 2:
                        continue
                    next_occupancy = dict(occupancy)
                    for point in moved_shape:
                        next_occupancy.setdefault(point, point in moved_green)
                    visit(rest, next_occupancy, next_counts,
                          {**placements, index: (turns, dr, dc)})

    visit((1, 2, 3, 4), occupancy, green_counts, {0: (0, 0, 0)})
    return solutions


if __name__ == "__main__":
    found = search()
    print("solutions", len(found))
    for solution in found:
        all_cells = set()
        for index, (turns, row, col) in solution.items():
            all_cells.update(shift(ORIENTATIONS[index][turns][0], row, col))
        bounds = (
            min(row for row, _ in all_cells),
            min(col for _, col in all_cells),
            max(row for row, _ in all_cells),
            max(col for _, col in all_cells),
        )
        print("layout", solution, "bounds", bounds, "area", len(all_cells))
