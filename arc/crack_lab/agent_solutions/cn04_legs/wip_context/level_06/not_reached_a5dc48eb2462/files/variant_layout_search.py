"""Pair every observed landmark using the true action-5 variants."""
from collections import Counter


GREENS = [
    [
        {(2, 4), (4, 2), (5, 0)},
        {(0, 0), (2, 1), (4, 3)},
        {(0, 4), (1, 2), (3, 0)},
        {(0, 2), (2, 4), (4, 5)},
    ],
    [
        {(2, 0), (3, 2)},
        {(0, 1), (2, 0)},
        {(0, 1), (1, 3)},
        {(1, 3), (3, 2)},
    ],
    [
        {(0, 2), (2, 0)},
        {(0, 2), (2, 0), (5, 1)},
        {(0, 2), (2, 0), (5, 1)},
        {(0, 2), (2, 0), (5, 1), (5, 6)},
        {(0, 2), (2, 0), (5, 1), (5, 6)},
    ],
    [
        {(0, 0), (5, 1)},
        {(0, 5), (1, 0)},
        {(0, 0), (5, 1)},
        {(0, 5), (1, 0)},
    ],
    [
        {(0, 2), (0, 4), (3, 0)},
        {(0, 2), (0, 5), (3, 0)},
        {(0, 2), (0, 6), (3, 0)},
        {(0, 2), (0, 7), (3, 0)},
    ],
]


def moved(points, row, col):
    return {(r + row, c + col) for r, c in points}


def search(limit=200):
    solutions = []
    seen = set()

    def visit(remaining, counts, owners, edges, placements):
        key = (
            tuple(sorted(remaining)),
            tuple(sorted(owners.items())),
            tuple(sorted(edges)),
        )
        if key in seen:
            return
        seen.add(key)
        if not remaining:
            if set(counts.values()) == {2}:
                solutions.append(dict(placements))
            return
        unmatched = {point for point, amount in counts.items() if amount == 1}
        endpoints_left = max(
            sum(max(len(green) for green in GREENS[index])
                for index in remaining),
            1,
        )
        if len(unmatched) > endpoints_left:
            return
        for index in remaining:
            rest = tuple(i for i in remaining if i != index)
            unique = set()
            for turns, green in enumerate(GREENS[index]):
                signature = tuple(sorted(green))
                if signature in unique:
                    continue
                unique.add(signature)
                candidates = set()
                for source in green:
                    for target in unmatched:
                        candidates.add((target[0] - source[0],
                                        target[1] - source[1]))
                for row, col in candidates:
                    placed = moved(green, row, col)
                    next_counts = counts.copy()
                    next_counts.update(placed)
                    if max(next_counts.values()) > 2:
                        continue
                    next_owners = dict(owners)
                    next_edges = set(edges)
                    legal = True
                    for point in placed:
                        prior = next_owners.get(point, ())
                        for other in prior:
                            edge = tuple(sorted((index, other)))
                            if edge in next_edges:
                                legal = False
                                break
                            next_edges.add(edge)
                        if not legal:
                            break
                        next_owners[point] = tuple(sorted(prior + (index,)))
                    if not legal:
                        continue
                    remaining_capacity = sum(
                        max(len(choice) for choice in GREENS[item])
                        for item in rest
                    )
                    if sum(value == 1 for value in next_counts.values()) > remaining_capacity:
                        continue
                    visit(
                        rest,
                        next_counts,
                        next_owners,
                        next_edges,
                        {**placements, index: (turns, row, col)},
                    )
                    if len(solutions) >= limit:
                        return

    for anchor_turns, anchor_green in enumerate(GREENS[0]):
        counts = Counter(anchor_green)
        owners = {point: (0,) for point in anchor_green}
        visit(
            (1, 2, 3, 4),
            counts,
            owners,
            set(),
            {0: (anchor_turns, 0, 0)},
        )
    return solutions, len(seen)


if __name__ == "__main__":
    found, states = search()
    ranked = []
    for solution in found:
        points = set()
        turns = 0
        for index, (count, row, col) in solution.items():
            turns += count
            points.update(moved(GREENS[index][count], row, col))
        bounds = (
            min(row for row, _ in points),
            min(col for _, col in points),
            max(row for row, _ in points),
            max(col for _, col in points),
        )
        span = bounds[2] - bounds[0] + bounds[3] - bounds[1]
        ranked.append((turns, span, bounds, solution))
    ranked.sort(key=lambda item: item[:3])
    print("states", states, "solutions", len(found))
    for item in ranked[:30]:
        print(item)
