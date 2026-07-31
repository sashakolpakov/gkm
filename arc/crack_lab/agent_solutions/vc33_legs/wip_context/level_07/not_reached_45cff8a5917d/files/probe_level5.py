import json
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import arr, color_counts, connected_components, frame_delta


def blobs(frame, min_area=2):
    return [
        (b.color, b.bbox, b.area)
        for b in connected_components(frame, min_area=min_area)
        if b.area < 1800
    ]


def inspect(env):
    with open("checkpoint.json") as stream:
        path = json.load(stream)["final_path"]
    for action in path:
        env.step(*action)
    print("LEVEL", env.levels_completed, "ACTIONS", env.actions)
    print("COUNTS", color_counts(env.frame()))
    print("BLOBS", blobs(env.frame()))

    base = arr(env.frame()).copy()
    candidates = connected_components(env.frame(), min_area=2)
    for index, blob in enumerate(candidates):
        if blob.area >= 1800:
            continue
        x, y = int(blob.centroid[1]), int(blob.centroid[0])
        clone = env.clone()
        clone.step(6, x, y)
        delta = frame_delta(base, clone.frame())
        if delta["count"] or clone.levels_completed != env.levels_completed:
            print(
                "CLICK",
                index,
                (blob.color, blob.bbox, blob.area),
                (x, y),
                "LEVEL",
                clone.levels_completed,
                "DELTA",
                (delta["count"], delta["bbox"]),
                "AFTER",
                blobs(clone.frame()),
            )

    staged = env.clone()
    full_path = []

    def pair(node, color):
        marks = connected_components(node.frame(), colors=(color,), min_area=2)
        if len(marks) != 2 or marks[0].area == marks[1].area:
            return None
        return max(marks, key=lambda b: b.area), min(marks, key=lambda b: b.area)

    def progress(node, color):
        found = pair(node, color)
        if found is None:
            return None
        moving, fixed = found
        lo, hi = sorted((moving.centroid[0], fixed.centroid[0]))
        gates = [
            b for b in connected_components(
                node.frame(), colors=(1, 12, 13, 14, 15), min_area=12
            )
            if lo < b.centroid[0] < hi and b.size[1] > b.size[0]
        ]
        return len(gates), abs(moving.centroid[0] - fixed.centroid[0]), abs(
            moving.centroid[1] - fixed.centroid[1]
        )

    def options(node):
        out = [
            (6, int(b.centroid[1]), int(b.centroid[0]))
            for b in connected_components(node.frame(), colors=(9,), min_area=2)
        ]
        out.extend(
            (6, int(b.centroid[1]), int(b.centroid[0]))
            for b in connected_components(
                node.frame(), colors=(12, 13, 14, 15), min_area=12
            )
            if b.size[1] > b.size[0]
        )
        return out

    for color in (11, 14):
        for stage in range(14):
            initial = progress(staged, color)
            if initial is None or staged.levels_completed > env.levels_completed:
                break

            def improved(node):
                if node.levels_completed > env.levels_completed:
                    return True
                current = progress(node, color)
                if current is None:
                    return False
                if initial[0]:
                    return current[0] < initial[0]
                return current[0] == 0 and current[2] < initial[2]

            queue = deque([([], staged.clone())])
            seen = {arr(staged.frame())[1:].tobytes()}
            found = None
            while queue and len(seen) <= 2500:
                path, node = queue.popleft()
                if len(path) >= 24:
                    continue
                for action in options(node):
                    child = node.clone()
                    child.step(*action)
                    child_path = path + [action]
                    if improved(child):
                        found = child_path
                        queue.clear()
                        break
                    key = arr(child.frame())[1:].tobytes()
                    if key not in seen:
                        seen.add(key)
                        queue.append((child_path, child))
            print("STAGE", color, stage, initial, len(seen), found)
            if not found:
                break
            for action in found:
                staged.step(*action)
            full_path.extend(found)
    print(
        "STAGED_RESULT",
        staged.levels_completed,
        progress(staged, 14),
        progress(staged, 11),
        full_path,
    )


levels, path, err = A.run_program("vc33", inspect)
print("DONE", levels, len(path), err)
