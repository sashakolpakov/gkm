import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import color_counts, connected_components, frame_delta
from legs import stage_large_square_for_diagonal_partner


def summary(frame):
    blobs = connected_components(frame, min_area=4)
    return {
        "colors": color_counts(frame),
        "blobs": [
            (b.color, b.bbox, b.area)
            for b in blobs
            if b.bbox[0] >= 8 or b.area < 1000
        ],
    }


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
        groups.append((
            round(sum(row for row, _ in group) / len(group)),
            round(sum(col for _, col in group) / len(group)),
            len(group),
        ))
    return sorted(groups)


def compact(frame):
    pieces = [
        (b.color, b.bbox, b.area)
        for b in connected_components(frame, min_area=1)
        if b.bbox[0] >= 10
        and b.size[0] == b.size[1]
        and b.area == b.size[0] ** 2
        and b.color not in (3, 4, 5, 7, 9)
    ]
    rings = [
        (b.bbox, b.area)
        for b in connected_components(frame, colors=(9,), min_area=9)
        if b.bbox[0] >= 10
    ]
    return {"pieces": pieces, "bodies": bodies(frame), "rings": rings}


def inspect(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(*action)
    base = env.frame()
    print("level", env.levels_completed + 1, "actions", env.actions)
    print("initial", summary(base))
    print("compact", compact(base))

    probes = []
    blobs = connected_components(base, min_area=4)
    for blob in blobs:
        if blob.bbox[0] >= 8 and blob.area < 1000:
            row, col = blob.centroid
            probes.append((6, int(round(col)), int(round(row))))
    probes.extend([(6, 0, 0), (6, 32, 32), (6, 63, 63)])

    seen = set()
    for action in probes:
        if action in seen:
            continue
        seen.add(action)
        clone = env.clone()
        clone.step(*action)
        delta = frame_delta(base, clone.frame())
        print("probe", action, "level", clone.levels_completed,
              "delta", (delta["count"], delta["bbox"]),
              "state", compact(clone.frame()))

    clone = env.clone()
    stage_large_square_for_diagonal_partner(clone, max_moves=20)
    print("existing_stage", "level", clone.levels_completed,
          "state", compact(clone.frame()))


A.run_program("su15", inspect)
