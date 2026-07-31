from collections import Counter


SHAPES = {
    "top": [
        {(2, 18), (3, 16)},
        {(2, 19), (4, 20)},
        {(5, 18), (6, 16)},
        {(2, 16), (4, 17)},
    ],
    "central": [
        {(5, 10), (6, 8), (7, 11)},
        {(5, 10), (6, 8), (7, 11), (8, 8)},
        {(5, 10), (6, 8), (7, 11), (8, 8), (9, 13)},
        {(5, 10), (6, 8), (7, 11), (8, 8), (9, 13), (10, 8)},
        {(5, 10), (6, 8), (7, 11), (8, 8), (9, 13), (10, 8),
         (11, 12)},
        {(5, 10), (6, 8), (7, 11), (8, 8), (9, 13), (10, 8)},
        {(5, 10), (6, 8), (7, 11), (8, 8), (9, 13)},
        {(5, 10), (6, 8), (7, 11), (8, 8)},
    ],
    "left": [
        {(13, 4), (15, 4), (17, 4)},
        {(15, 2), (15, 4), (15, 6)},
        {(13, 2), (15, 2), (17, 2)},
        {(13, 2), (13, 4), (13, 6)},
    ],
    "right": [
        {(16, 16), (17, 18)},
        {(16, 18), (18, 17)},
        {(17, 16), (18, 18)},
        {(16, 17), (18, 16)},
    ],
}


def shift(points, dy, dx):
    return frozenset((r + dy, c + dx) for r, c in points)


solutions = []


def extend(placed, remaining, counts):
    if not remaining:
        if counts and all(n == 2 for n in counts.values()):
            cost = sum(abs(dy) + abs(dx)
                       for _, _, dy, dx, _ in placed)
            solutions.append((cost, tuple(placed)))
        return
    anchors = tuple(counts)
    for name in remaining:
        poses = set()
        for turns, raw in enumerate(SHAPES[name]):
            for ar, ac in anchors:
                for er, ec in raw:
                    dy, dx = ar - er, ac - ec
                    pose = shift(raw, dy, dx)
                    if all(0 <= r <= 20 and 0 <= c <= 21 for r, c in pose):
                        poses.add((turns, dy, dx, pose))
        for turns, dy, dx, pose in poses:
            nxt = counts.copy()
            nxt.update(pose)
            if max(nxt.values()) > 2:
                continue
            extend(placed + [(name, turns, dy, dx, pose)],
                   tuple(n for n in remaining if n != name), nxt)


for top_turns, raw_top in enumerate(SHAPES["top"]):
    top = frozenset(raw_top)
    extend([("top", top_turns, 0, 0, top)],
           ("central", "left", "right"), Counter(top))
unique = {}
for cost, placed in solutions:
    key = tuple(sorted((name, turns, dy, dx)
                       for name, turns, dy, dx, _ in placed))
    unique[key] = min(cost, unique.get(key, cost))
print("COUNT", len(unique))
for key, cost in sorted(unique.items(), key=lambda item: (item[1], item[0]))[:100]:
    print(cost, key)
