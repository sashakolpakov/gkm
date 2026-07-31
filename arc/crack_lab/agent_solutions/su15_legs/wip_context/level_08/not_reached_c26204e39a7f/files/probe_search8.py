import heapq
import itertools
import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import connected_components


def bodies(frame):
    points = {
        (row, col)
        for row in range(10, 64)
        for col in range(64)
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


def squares(frame):
    return [
        blob for blob in connected_components(frame, colors=(8, 11), min_area=1)
        if blob.bbox[0] >= 10
        and blob.size[0] == blob.size[1]
        and blob.area == blob.size[0] ** 2
        and blob.color in (8, 11)
    ]


def center(blob):
    return round(blob.centroid[0]), round(blob.centroid[1])


def inspect(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(*action)
    start_level = env.levels_completed
    initial = env.frame()
    targets = tuple(
        center(blob)
        for blob in connected_components(initial, colors=(9,), min_area=9)
        if blob.bbox[0] >= 10
    )

    def state(node):
        frame = node.frame()
        sq = squares(frame)
        agents = bodies(frame)
        return sq, agents

    def key(node):
        sq, agents = state(node)
        return (
            tuple(sorted((blob.color, blob.bbox) for blob in sq)),
            agents,
            tuple(
                int(node.frame()[row][col])
                for tr, tc in targets
                for row in range(tr - 4, tr + 5)
                for col in range(tc - 4, tc + 5)
            ),
        )

    def assignment_distance(node):
        sq, agents = state(node)
        large = [center(blob) for blob in sq if blob.color == 8]
        entities = large + list(agents)
        if len(entities) != 4:
            return 10000
        return min(
            sum(max(abs(er - tr), abs(ec - tc))
                for (er, ec), (tr, tc) in zip(entities, order))
            for order in itertools.permutations(targets)
        )

    def progress(node):
        frame = node.frame()
        return sum(
            sum(int(frame[row][col]) != 9
                for row in range(tr - 4, tr + 5)
                for col in range(tc - 4, tc + 5))
            for tr, tc in targets
        )

    def actions(node):
        sq, agents = state(node)
        points = []
        entities = list(agents) + [center(blob) for blob in sq]
        for row, col in entities:
            points.append((6, col, row))
            for target_row, target_col in targets:
                for step in (4, 6, 8):
                    next_row = row + max(-step, min(step, target_row - row))
                    next_col = col + max(-step, min(step, target_col - col))
                    points.append((6, next_col, next_row))
        points.extend((6, col, row) for row, col in targets)
        points.extend(((6, 4, 13), (6, 59, 13), (6, 4, 59), (6, 59, 59)))
        return list(dict.fromkeys(points))

    serial = itertools.count()
    root = env.clone()
    heap = [(assignment_distance(root), 0, next(serial), root, [])]
    seen = {key(root)}
    best = None
    for expanded in range(400):
        if not heap:
            break
        _, depth, _, node, path = heapq.heappop(heap)
        metric = (assignment_distance(node), -progress(node))
        if best is None or metric < best:
            best = metric
            print("best", expanded, depth, metric, state(node)[1],
                  [(b.color, b.bbox) for b in state(node)[0]], "path", path)
        if depth >= 24:
            continue
        for action in actions(node):
            child = node.clone()
            child.step(*action)
            child_path = path + [action]
            if child.levels_completed > start_level:
                print("FOUND", child_path)
                return
            if child.terminal():
                continue
            child_key = key(child)
            if child_key in seen:
                continue
            seen.add(child_key)
            child_depth = len(child_path)
            score = assignment_distance(child) + child_depth * 0.2 - progress(child) * 0.05
            heapq.heappush(
                heap, (score, child_depth, next(serial), child, child_path)
            )
    print("NO_PATH", "seen", len(seen), "best", best)


A.run_program("su15", inspect)
