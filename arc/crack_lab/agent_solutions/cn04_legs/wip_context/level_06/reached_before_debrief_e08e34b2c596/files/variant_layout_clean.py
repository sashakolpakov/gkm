"""Search true variants with only green-to-green physical overlaps."""
from collections import Counter

from variant_layout_search import GREENS, moved


SHAPES = [
    [
        {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 4),
         (2, 0), (2, 4), (3, 0), (4, 0), (4, 1), (4, 2), (5, 0)},
        {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 1),
         (1, 5), (2, 1), (2, 5), (3, 5), (4, 3), (4, 4), (4, 5)},
        {(0, 4), (1, 2), (1, 3), (1, 4), (2, 4), (3, 0), (3, 4),
         (4, 0), (4, 4), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4)},
        {(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 4), (3, 0),
         (3, 4), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5)},
    ],
    [
        {(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 3), (2, 0),
         (2, 3), (3, 2), (3, 3)},
        {(0, 1), (0, 2), (0, 3), (1, 3), (2, 0), (2, 3), (3, 0),
         (3, 1), (3, 2), (3, 3)},
        {(0, 0), (0, 1), (1, 0), (1, 3), (2, 0), (2, 3), (3, 0),
         (3, 1), (3, 2), (3, 3)},
        {(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 3), (2, 0),
         (3, 0), (3, 1), (3, 2)},
    ],
    [
        {(0, 2), (1, 1), (1, 2), (2, 0), (2, 1), (3, 1), (4, 1)},
        {(0, 2), (1, 1), (1, 2), (2, 0), (2, 1), (3, 1), (4, 1),
         (4, 2), (4, 3), (5, 1)},
        {(0, 2), (1, 1), (1, 2), (2, 0), (2, 1), (3, 1), (4, 1),
         (4, 2), (4, 3), (4, 4), (4, 5), (5, 1)},
        {(0, 2), (1, 1), (1, 2), (2, 0), (2, 1), (3, 1), (3, 6),
         (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (5, 1),
         (5, 6)},
        {(0, 2), (1, 1), (1, 2), (1, 6), (2, 0), (2, 1), (2, 6),
         (3, 1), (3, 6), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5),
         (4, 6), (5, 1), (5, 6)},
        {(0, 2), (0, 5), (0, 6), (1, 1), (1, 2), (1, 6), (1, 7),
         (2, 0), (2, 1), (2, 6), (3, 1), (3, 6), (4, 1), (4, 2),
         (4, 3), (4, 4), (4, 5), (4, 6), (5, 1), (5, 6)},
    ],
    [
        {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (5, 1)},
        {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 0)},
        {(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)},
        {(0, 5), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5)},
    ],
    [
        {(0, 2), (0, 4), (1, 2), (1, 4), (2, 1), (2, 2), (2, 4),
         (2, 5), (3, 0), (3, 1), (3, 5), (4, 1), (4, 2), (4, 3),
         (4, 4), (4, 5)},
        {(0, 2), (0, 5), (1, 2), (1, 5), (2, 1), (2, 2), (2, 5),
         (2, 6), (3, 0), (3, 1), (3, 6), (4, 1), (4, 2), (4, 3),
         (4, 4), (4, 5), (4, 6)},
        {(0, 2), (0, 6), (1, 2), (1, 6), (2, 1), (2, 2), (2, 6),
         (2, 7), (3, 0), (3, 1), (3, 7), (4, 1), (4, 2), (4, 3),
         (4, 4), (4, 5), (4, 6), (4, 7)},
        {(0, 2), (0, 7), (1, 2), (1, 7), (2, 1), (2, 2), (2, 7),
         (2, 8), (3, 0), (3, 1), (3, 8), (4, 1), (4, 2), (4, 3),
         (4, 4), (4, 5), (4, 6), (4, 7), (4, 8)},
    ],
]


def search(limit=100):
    solutions = []
    seen = set()

    def visit(remaining, green_counts, occupied, placements):
        key = (
            tuple(sorted(remaining)),
            tuple(sorted(green_counts.items())),
            tuple(sorted(occupied.items())),
        )
        if key in seen:
            return
        seen.add(key)
        if not remaining:
            if set(green_counts.values()) == {2}:
                solutions.append(dict(placements))
            return
        unmatched = {p for p, count in green_counts.items() if count == 1}
        capacity = sum(max(len(g) for g in GREENS[i]) for i in remaining)
        if len(unmatched) > capacity:
            return
        for index in remaining:
            rest = tuple(i for i in remaining if i != index)
            for turns, (shape, green) in enumerate(
                zip(SHAPES[index], GREENS[index])
            ):
                candidates = {
                    (target[0] - source[0], target[1] - source[1])
                    for source in green
                    for target in unmatched
                }
                for row, col in candidates:
                    placed_shape = moved(shape, row, col)
                    placed_green = moved(green, row, col)
                    overlaps = placed_shape & set(occupied)
                    if not overlaps:
                        continue
                    if not overlaps <= placed_green:
                        continue
                    if not all(occupied[point] for point in overlaps):
                        continue
                    next_counts = green_counts.copy()
                    next_counts.update(placed_green)
                    if max(next_counts.values()) > 2:
                        continue
                    next_occupied = dict(occupied)
                    for point in placed_shape:
                        next_occupied.setdefault(point, point in placed_green)
                    visit(
                        rest,
                        next_counts,
                        next_occupied,
                        {**placements, index: (turns, row, col)},
                    )
                    if len(solutions) >= limit:
                        return

    for anchor_turns, (anchor_shape, anchor_green) in enumerate(
        zip(SHAPES[0], GREENS[0])
    ):
        green_counts = Counter(anchor_green)
        occupied = {
            point: point in anchor_green for point in anchor_shape
        }
        visit(
            (1, 2, 3, 4),
            green_counts,
            occupied,
            {0: (anchor_turns, 0, 0)},
        )
    return solutions, len(seen)


if __name__ == "__main__":
    found, states = search()
    ranked = []
    for solution in found:
        all_cells = set()
        for index, (turns, row, col) in solution.items():
            all_cells.update(moved(SHAPES[index][turns], row, col))
        bounds = (
            min(row for row, _ in all_cells),
            min(col for _, col in all_cells),
            max(row for row, _ in all_cells),
            max(col for _, col in all_cells),
        )
        ranked.append((
            sum(turns for turns, _, _ in solution.values()),
            bounds[2] - bounds[0] + bounds[3] - bounds[1],
            bounds,
            solution,
        ))
    ranked.sort(key=lambda item: item[:3])
    print("states", states, "solutions", len(found))
    for item in ranked[:30]:
        print(item)
