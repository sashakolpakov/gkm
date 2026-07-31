import json

import gkm_try as H

from perception import color_counts, connected_components, frame_delta
from legs import (
    merge_equal_squares_and_deliver_to_ring,
    stage_large_square_for_diagonal_partner,
)


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
    merge_equal_squares_and_deliver_to_ring(clone, max_moves=32)
    print("existing_delivery", "level", clone.levels_completed,
          "state", compact(clone.frame()))
    for index in range(4):
        finals = [
            blob for blob in connected_components(
                clone.frame(), colors=(12,), min_area=9
            )
            if blob.bbox[0] >= 10
        ]
        targets = [
            blob for blob in connected_components(
                clone.frame(), colors=(9,), min_area=9
            )
            if blob.bbox[0] >= 10
        ]
        if not finals or not targets or clone.levels_completed != env.levels_completed:
            break
        final = finals[0]
        target = min(
            targets,
            key=lambda ring: (
                (ring.centroid[0] - final.centroid[0]) ** 2
                + (ring.centroid[1] - final.centroid[1]) ** 2
            ),
        )
        row = final.centroid[0] + max(
            -6, min(6, round(target.centroid[0] - final.centroid[0]))
        )
        col = final.centroid[1] + max(
            -6, min(6, round(target.centroid[1] - final.centroid[1]))
        )
        action = (6, round(col), round(row))
        clone.step(*action)
        print("deliver_color12", index + 1, "level", clone.levels_completed,
              "action", action, "state", compact(clone.frame()))
    for index in range(8):
        finals = [
            blob for blob in connected_components(
                clone.frame(), colors=(12,), min_area=9
            )
            if blob.bbox[0] >= 10
            and blob.size[0] == blob.size[1]
            and blob.area == blob.size[0] ** 2
        ]
        if len(finals) != 2:
            break
        first, second = sorted(finals, key=lambda blob: blob.centroid)
        fr, fc = first.centroid
        sr, sc = second.centroid
        distance = max(abs(fr - sr), abs(fc - sc))
        if distance <= 12:
            row, col = round((fr + sr) / 2), round((fc + sc) / 2)
        else:
            row = sr + max(-6, min(6, fr - sr))
            col = sc + max(-6, min(6, fc - sc))
        action = (6, round(col), round(row))
        clone.step(*action)
        print("merge_color12", index + 1, "level", clone.levels_completed,
              "action", action, "state", compact(clone.frame()))

    clone = env.clone()
    stage_large_square_for_diagonal_partner(clone, max_moves=20)
    print("existing_stage", "level", clone.levels_completed,
          "state", compact(clone.frame()))


H.A.run_program("su15", inspect)
