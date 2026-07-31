import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr, connected_components


SEED = (
    [3] * 8
    + [2, 4, 4, 3]
    + [1] * 9
    + [3] * 3
    + [1]
    + [3] * 2
    + [1]
)


def selected_shape(node):
    before = arr(node.frame()).copy()
    cursor = list(zip(*((before == 0).nonzero())))
    if len(cursor) != 1:
        return None
    center = tuple(int(value) for value in cursor[0])
    background = 5
    votes = []
    changed_from = set()
    for action in (1, 2, 3, 4):
        moved = node.clone()
        moved.step(action)
        after = arr(moved.frame())
        for row, col in zip(*((before != after).nonzero())):
            if int(after[row, col]) == background:
                value = int(before[row, col])
                if value not in (0, background):
                    votes.append(value)
        changed_from.update(
            (int(row), int(col), int(before[row, col]))
            for row, col in zip(*((before != after).nonzero()))
        )
    if not votes:
        return center, None
    color = max(set(votes), key=votes.count)
    points = {(row, col) for row, col, value in changed_from if value == color}
    rows = [row for row, _ in points]
    cols = [col for _, col in points]
    return center, color, len(points), (min(rows), min(cols), max(rows), max(cols))


def step_many(node, path):
    for action in path:
        node.step(action)


def probe(env):
    with open("checkpoint.json") as handle:
        step_many(env, json.load(handle)["final_path"])

    frame = arr(env.frame())
    barriers = [
        (blob.area, blob.bbox, blob.size)
        for blob in connected_components(frame, colors=(2,), min_area=1)
    ]
    print("barriers", barriers)
    print("initial", selected_shape(env))

    node = env.clone()
    node.step(5)
    step_many(node, [1] * 22)
    node.step(5)
    print("isolated", selected_shape(node))
    step_many(node, SEED)
    print("seed", len(SEED), selected_shape(node))
    placed = node.clone()
    step_many(placed, [2] * 10)
    print("first_placed", selected_shape(placed), "level", placed.levels_completed)
    placed.step(5)
    print("second_after_park", selected_shape(placed))

    tests = {
        "to_11_station": [1] * 3 + [3] * 2,
        "down_to_target": [2] * 10,
        "down_left": [2] * 10 + [3],
        "left_then_down": [3] * 2 + [2] * 10,
    }
    for name, path in tests.items():
        scout = node.clone()
        before = selected_shape(scout)
        step_many(scout, path)
        print(name, before, "=>", selected_shape(scout), "level", scout.levels_completed)


if __name__ == "__main__":
    print("RUN", A.run_program("re86", probe))
