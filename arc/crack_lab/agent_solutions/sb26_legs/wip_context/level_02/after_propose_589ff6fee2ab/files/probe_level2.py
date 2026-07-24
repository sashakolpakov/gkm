import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import action_deltas, color_counts, connected_components, frame_delta
from legs import copy_visible_color_code


def probe(env):
    with open("checkpoint.json") as f:
        checkpoint = json.load(f)
    for action in checkpoint["final_path"]:
        env.step(action)

    frame = env.frame()
    blobs = connected_components(frame, min_area=4)
    print("level", env.levels_completed + 1, "actions", env.actions)
    print("colors", color_counts(frame))
    print(
        "blobs",
        [
            (b.color, b.bbox, b.area, tuple(round(x, 1) for x in b.centroid))
            for b in blobs
        ],
    )
    print(
        "bare_deltas",
        {
            action: (delta["count"], delta["bbox"])
            for action, delta in action_deltas(env, env.actions).items()
        },
    )
    clone = env.clone()
    try:
        copy_visible_color_code(clone)
        print(
            "existing_leg",
            "levels",
            clone.levels_completed,
            "terminal",
            clone.terminal(),
            "delta",
            (lambda d: (d["count"], d["bbox"]))(frame_delta(frame, clone.frame())),
        )
    except Exception as exc:
        print("existing_leg_error", type(exc).__name__, str(exc))
    chars = {0: "0", 2: "2", 4: ".", 8: "8", 14: "E"}
    print("central_crop")
    for row in frame[15:43, 15:49]:
        print("".join(chars.get(int(value), "?") for value in row))
    palette = [
        b for b in blobs
        if b.bbox[0] >= 54 and b.area == 16 and b.color != 5
    ]
    slots = [(22, 22), (22, 28), (22, 34), (22, 40),
             (36, 22), (36, 28), (36, 34), (36, 40)]
    for swatch in palette:
        clone = env.clone()
        x = round(swatch.centroid[1])
        y = round(swatch.centroid[0])
        clone.step(6, x, y)
        select_delta = frame_delta(frame, clone.frame())
        clone.step(6, slots[0][1], slots[0][0])
        place_delta = frame_delta(frame, clone.frame())
        print(
            "place",
            swatch.color,
            "select",
            (select_delta["count"], select_delta["bbox"]),
            "then",
            (place_delta["count"], place_delta["bbox"]),
            "slot_color",
            int(clone.frame()[slots[0]]),
        )
    palette_xy = {
        b.color: (round(b.centroid[1]), round(b.centroid[0]))
        for b in palette
    }

    def try_pattern(name, colors):
        clone = env.clone()
        for (row, col), color in zip(slots, colors):
            if color == 14 and (row, col) == (22, 34):
                continue
            clone.step(6, *palette_xy[color])
            clone.step(6, col, row)
        before_submit = clone.frame().copy()
        clone.step(5)
        delta = frame_delta(before_submit, clone.frame())
        print(
            "pattern",
            name,
            colors,
            "levels",
            clone.levels_completed,
            "submit_delta",
            (delta["count"], delta["bbox"]),
        )

    try_pattern("literal_skip", [12, 15, 14, 8, 9, 14, 11, 6])
    try_pattern("cycle_forward", [8, 9, 14, 11, 6, 12, 15, 8])
    try_pattern("cycle_reverse", [6, 11, 14, 9, 8, 15, 12, 6])
    try_pattern("cycle_forward_snake", [8, 9, 14, 11, 8, 15, 12, 6])
    try_pattern("cycle_reverse_snake", [6, 11, 14, 9, 6, 12, 15, 8])
    try_pattern("rows_start_at_border", [8, 9, 14, 11, 14, 11, 6, 12])
    target = [12, 15, 8, 9, 14, 11, 6]
    blank_indices = [0, 1, 3, 4, 5, 6, 7]
    top = [0, 1, 3]
    bottom = [4, 5, 6, 7]
    for first_name, first, second_name, second in (
        ("top_lr", top, "bottom_rl", bottom[::-1]),
        ("top_rl", top[::-1], "bottom_lr", bottom),
        ("top_rl", top[::-1], "bottom_rl", bottom[::-1]),
        ("bottom_lr", bottom, "top_lr", top),
        ("bottom_lr", bottom, "top_rl", top[::-1]),
        ("bottom_rl", bottom[::-1], "top_lr", top),
        ("bottom_rl", bottom[::-1], "top_rl", top[::-1]),
    ):
        assignment = [None] * 8
        assignment[2] = 14
        for index, color in zip(first + second, target):
            assignment[index] = color
        try_pattern(first_name + "_" + second_name, assignment)


A.run_program("sb26", probe)
