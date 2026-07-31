"""Search layouts where every green cell is covered and bodies do not clash."""
from layout_search_clean import ORIENTATIONS, shift


def search(limit=10):
    shape, green = ORIENTATIONS[0][0]
    occupancy = {point: [point in green] for point in shape}
    solutions = []
    seen = set()

    def visit(remaining, occupancy, placements):
        if len(solutions) >= limit:
            return
        key = (
            remaining,
            tuple(sorted((point, tuple(sorted(flags)))
                         for point, flags in occupancy.items())),
        )
        if key in seen:
            return
        seen.add(key)
        if not remaining:
            if all(
                len(occupancy[point]) >= 2
                for point, flags in occupancy.items()
                if any(flags)
            ):
                solutions.append(dict(placements))
            return
        open_green = {
            point for point, flags in occupancy.items()
            if any(flags) and len(flags) == 1
        }
        occupied_points = set(occupancy)
        for index in remaining[:1]:
            rest = tuple(i for i in remaining if i != index)
            for turns, (local_shape, local_green) in enumerate(ORIENTATIONS[index]):
                candidates = set()
                for source in local_green:
                    for target in occupied_points:
                        candidates.add((target[0] - source[0],
                                        target[1] - source[1]))
                for source in local_shape:
                    for target in open_green:
                        candidates.add((target[0] - source[0],
                                        target[1] - source[1]))
                for dr, dc in candidates:
                    moved_shape = shift(local_shape, dr, dc)
                    moved_green = shift(local_green, dr, dc)
                    combined = occupied_points | moved_shape
                    if (
                        max(row for row, _ in combined)
                        - min(row for row, _ in combined) > 11
                        or max(col for _, col in combined)
                        - min(col for _, col in combined) > 11
                    ):
                        continue
                    overlaps = moved_shape & occupied_points
                    if not overlaps:
                        continue
                    valid = True
                    for point in overlaps:
                        incoming_green = point in moved_green
                        if not incoming_green and not any(occupancy[point]):
                            valid = False
                            break
                    if not valid:
                        continue
                    next_occupancy = {
                        point: list(flags) for point, flags in occupancy.items()
                    }
                    for point in moved_shape:
                        next_occupancy.setdefault(point, []).append(
                            point in moved_green
                        )
                    visit(rest, next_occupancy,
                          {**placements, index: (turns, dr, dc)})

    visit((1, 2, 3, 4), occupancy, {0: (0, 0, 0)})
    return solutions, len(seen)


if __name__ == "__main__":
    found, states = search()
    print("states", states, "solutions", len(found))
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
