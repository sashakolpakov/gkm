import heapq
import itertools
import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import connected_components


PREFIX = [
    (6, 49, 29), (6, 49, 50), (6, 53, 23), (6, 15, 19),
    (6, 49, 50), (6, 7, 19), (6, 49, 19), (6, 7, 50),
    (6, 47, 55), (6, 56, 19), (6, 52, 56), (6, 7, 49),
    (6, 52, 55), (6, 53, 19), (6, 7, 53), (6, 56, 55),
    (6, 4, 59), (6, 53, 19), (6, 53, 52),
]


def centers(frame, color):
    points = {
        (row, col)
        for row in range(10, 64)
        for col in range(64)
        if int(frame[row][col]) == color
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
                if max(abs(point[0] - other[0]), abs(point[1] - other[1])) <= 1
            }
            points -= near
            todo.extend(near)
        if len(group) >= 4:
            groups.append((
                round(sum(row for row, _ in group) / len(group)),
                round(sum(col for _, col in group) / len(group)),
            ))
    return tuple(sorted(groups))


def inspect(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(*action)
    start_level = env.levels_completed
    initial = env.frame()
    targets = tuple(
        (round(blob.centroid[0]), round(blob.centroid[1]))
        for blob in connected_components(initial, colors=(9,), min_area=9)
        if blob.bbox[0] >= 10
    )
    root = env.clone()
    for action in PREFIX:
        root.step(*action)

    def state(node):
        frame = node.frame()
        bodies = centers(frame, 7)
        square = tuple(
            (blob.color, blob.bbox)
            for blob in connected_components(frame, colors=(8,), min_area=9)
            if blob.bbox[0] >= 10
        )
        return square, bodies

    def key(node):
        square, bodies = state(node)
        frame = node.frame()
        pixels = tuple(
            (row, col)
            for row in range(10, 64)
            for col in range(64)
            if int(frame[row][col]) == 7
        )
        return square, pixels

    def distance(node):
        square, bodies = state(node)
        squares = [
            ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            for _, bbox in square
        ]
        entities = squares + list(bodies)
        if len(entities) != 4:
            return 999
        return min(
            sum(max(abs(er - tr), abs(ec - tc))
                for (er, ec), (tr, tc) in zip(entities, order))
            for order in itertools.permutations(targets)
        )

    print("root", distance(root), state(root), "terminal", root.terminal())
    serial = itertools.count()
    heap = [(distance(root), 0, next(serial), root, [])]
    seen = {key(root)}
    best = distance(root)
    for expanded in range(3000):
        if not heap:
            break
        _, depth, _, node, path = heapq.heappop(heap)
        if depth >= 24:
            continue
        frame = node.frame()
        actions = [
            (6, col, row)
            for row in range(10, 64)
            for col in range(64)
            if int(frame[row][col]) == 7
        ]
        actions.append((6, 32, 32))
        for action in actions:
            child = node.clone()
            child.step(*action)
            child_path = path + [action]
            if child.levels_completed > start_level:
                print("FOUND", PREFIX + child_path)
                return
            if child.terminal():
                continue
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            metric = distance(child)
            if metric < best:
                best = metric
                print("best", expanded, len(child_path), metric, state(child),
                      "suffix", child_path)
            heapq.heappush(
                heap,
                (metric + len(child_path) * 0.05,
                 len(child_path), next(serial), child, child_path),
            )
    print("NO_PATH", len(seen), best)


if __name__ == "__main__":
    A.run_program("su15", inspect)
