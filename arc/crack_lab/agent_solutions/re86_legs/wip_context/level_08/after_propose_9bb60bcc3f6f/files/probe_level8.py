import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import arr, action_deltas, color_counts, connected_components, frame_delta
from legs import _selection_cycle, _shape_offsets


def probe(env):
    with open("checkpoint.json") as handle:
        path = json.load(handle)["final_path"]
    for action in path:
        env.step(action)

    print("level", env.levels_completed, "actions", list(env.actions))
    frame = arr(env.frame())
    print("colors", color_counts(frame))
    for color in (1, 15):
        print(
            "components",
            color,
            [(blob.area, blob.bbox) for blob in connected_components(
                frame, colors=(color,), min_area=1
            )],
        )
    rings = [
        blob for blob in connected_components(frame, colors=(4,), min_area=8)
        if blob.size == (3, 3) and blob.area == 8
    ]
    print("rings", [(b.centroid, int(frame[int(b.centroid[0]), int(b.centroid[1])])) for b in rings])
    stations = [
        blob for blob in connected_components(frame, colors=(2,), min_area=8)
        if blob.size == (5, 5)
    ]
    print(
        "stations",
        [(b.bbox, int(frame[b.bbox[0] + 2, b.bbox[1] + 2])) for b in stations],
    )
    print("one_step")
    for action, delta in action_deltas(env, env.actions).items():
        print(action, delta["count"], delta["bbox"], delta["samples"][:8])
    node = env.clone()
    base = frame.copy()
    print("selection_cycle")
    for index in range(8):
        moved = node.clone()
        before = arr(moved.frame()).copy()
        moved.step(1)
        delta = frame_delta(before, moved.frame())
        changed = arr(before) != arr(moved.frame())
        old_colors = sorted(set(int(v) for v in before[changed]))
        new_colors = sorted(set(int(v) for v in arr(moved.frame())[changed]))
        print(index, delta["count"], delta["bbox"], old_colors, new_colors)
        node.step(5)
    print("eight_uses_delta", frame_delta(base, node.frame()))
    centers = _selection_cycle(env)
    print("centers", centers)
    for index in range(len(centers)):
        center, color, offsets = _shape_offsets(env, index, len(centers), 22)
        print(
            "shape",
            index,
            center,
            color,
            len(offsets),
            sorted(offsets),
        )
    scout = env.clone()
    scout.step(5)
    for _ in range(22):
        scout.step(2)
    scout.step(5)

    def moving_summary(node):
        before = arr(node.frame()).copy()
        center = tuple(int(v) for v in list(zip(*((before == 0).nonzero())))[0])
        frames = []
        votes = []
        background = 5
        for action in (1, 2, 3, 4):
            moved = node.clone()
            moved.step(action)
            after = arr(moved.frame())
            frames.append(after)
            for row, col in zip(*((before != after).nonzero())):
                if int(after[row, col]) == background and int(before[row, col]) not in (0, background):
                    votes.append(int(before[row, col]))
        if not votes:
            return center, None
        color = max(set(votes), key=votes.count)
        points = set()
        for after in frames:
            for row, col in zip(*((before != after).nonzero())):
                if int(before[row, col]) == color:
                    points.add((int(row) - center[0], int(col) - center[1]))
        rows = [p[0] for p in points]
        cols = [p[1] for p in points]
        return center, (
            color,
            len(points),
            (min(rows), min(cols), max(rows), max(cols)),
        )

    print("up_collision")
    for step_index in range(12):
        print(step_index, moving_summary(scout))
        scout.step(1)


A.run_program("re86", probe)
