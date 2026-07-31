import json

import gkm_try as H

from perception import color_counts, connected_components, frame_delta


PREFIX = [
    (6, 49, 29), (6, 49, 50), (6, 53, 23), (6, 15, 19),
    (6, 49, 50), (6, 7, 19), (6, 49, 19), (6, 7, 50),
    (6, 47, 55), (6, 56, 19), (6, 52, 56), (6, 7, 49),
    (6, 52, 55), (6, 53, 19), (6, 7, 53), (6, 56, 55),
    (6, 4, 59), (6, 53, 19), (6, 53, 52),
]


def body_pixels(frame):
    return {
        (row, col)
        for row in range(10, 64)
        for col in range(64)
        if int(frame[row][col]) == 7
    }


def body_groups(frame):
    remaining = body_pixels(frame)
    groups = []
    while remaining:
        todo = [remaining.pop()]
        group = []
        while todo:
            point = todo.pop()
            group.append(point)
            near = {
                other
                for other in remaining
                if max(abs(point[0] - other[0]), abs(point[1] - other[1])) <= 1
            }
            remaining -= near
            todo.extend(near)
        if len(group) >= 4:
            groups.append(tuple(sorted(group)))
    return tuple(sorted(groups))


def bodies(frame):
    return tuple(sorted(
        (
            round(sum(row for row, _ in group) / len(group)),
            round(sum(col for _, col in group) / len(group)),
            len(group),
        )
        for group in body_groups(frame)
    ))


def compact(frame):
    objects = [
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, min_area=4)
        if blob.bbox[0] >= 10 and blob.area < 1000
    ]
    return {
        "colors": color_counts(frame),
        "objects": objects,
        "bodies": bodies(frame),
    }


def inspect(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(*action)
    base_level = int(env.levels_completed)
    base = env.frame()
    ring_mask = {
        (row, col)
        for row in range(10, 64)
        for col in range(64)
        if int(base[row][col]) == 9
    }
    rings = [
        (blob.bbox, blob.area)
        for blob in connected_components(base, colors=(9,), min_area=9)
        if blob.bbox[0] >= 10
    ]
    square = [
        blob.bbox
        for blob in connected_components(base, colors=(8,), min_area=9)
        if blob.bbox[0] >= 10
    ]
    print("ROOT", base_level, repr(env.actions), "rings", rings,
          "square", square, "bodies", bodies(base))
    print("COUNTS", color_counts(base))

    probes = [(6, 32, 32)]
    probes += [
        (6, (bbox[1] + bbox[3]) // 2, (bbox[0] + bbox[2]) // 2)
        for bbox, _ in rings
    ]
    probes += [
        (6, col, row)
        for row, col, _ in bodies(base)
    ]
    for action in probes:
        clone = env.clone()
        clone.step(*action)
        delta = frame_delta(base, clone.frame())
        after_square = [
            blob.bbox
            for blob in connected_components(
                clone.frame(), colors=(8,), min_area=9
            )
            if blob.bbox[0] >= 10
        ]
        print("ACTION", action, "delta", (delta["count"], delta["bbox"]),
              "level", int(clone.levels_completed), "square", after_square,
              "bodies", bodies(clone.frame()))

    clone = env.clone()
    print("PREFIX 0", square, bodies(base), "overlap",
          len(body_pixels(base) & ring_mask))
    for index, action in enumerate(PREFIX, 1):
        clone.step(*action)
        if index in (1, 5, 10, 15, len(PREFIX)):
            after_square = [
                blob.bbox
                for blob in connected_components(
                    clone.frame(), colors=(8,), min_area=9
                )
                if blob.bbox[0] >= 10
            ]
            print("PREFIX", index, after_square, bodies(clone.frame()),
                  "overlap", len(body_pixels(clone.frame()) & ring_mask),
                  "level", int(clone.levels_completed))

    print("NEAR_SHAPES")
    for group in body_groups(clone.frame()):
        anchor_row = min(row for row, _ in group)
        anchor_col = min(col for _, col in group)
        relative = tuple(
            (row - anchor_row, col - anchor_col) for row, col in group
        )
        print(
            "BODY",
            (anchor_row, anchor_col),
            relative,
            "overlap",
            len(set(group) & ring_mask),
        )

    outcomes = {}
    for row, col in sorted(body_pixels(clone.frame())):
        child = clone.clone()
        child.step(6, col, row)
        key = (
            tuple(
                blob.bbox
                for blob in connected_components(
                    child.frame(), colors=(8,), min_area=9
                )
                if blob.bbox[0] >= 10
            ),
            body_groups(child.frame()),
        )
        outcomes.setdefault(key, []).append((6, col, row))
    ranked = sorted(
        outcomes.items(),
        key=lambda item: (
            -len(body_pixels_from_groups(item[0][1]) & ring_mask),
            item[0],
        ),
    )
    print("NEAR_OUTCOMES", len(outcomes))
    for key, actions in ranked[:12]:
        square_key, groups = key
        overlap = len(body_pixels_from_groups(groups) & ring_mask)
        centers = tuple(sorted(
            (
                round(sum(row for row, _ in group) / len(group)),
                round(sum(col for _, col in group) / len(group)),
            )
            for group in groups
        ))
        print("OUT", overlap, square_key, centers, "actions", actions)


def body_pixels_from_groups(groups):
    return {point for group in groups for point in group}


if __name__ == "__main__":
    H.A.run_program("su15", inspect)
