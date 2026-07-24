import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import action_deltas, color_counts, connected_components, frame_delta
from legs import copy_visible_color_code


def compact_blobs(frame):
    return [
        (b.color, b.bbox, b.area)
        for b in connected_components(frame, min_area=2)
    ]


def observe(env):
    verification = env.clone()
    copy_visible_color_code(verification)
    print(
        "leg_verification",
        verification.levels_completed,
        verification.terminal(),
    )
    print("level", env.levels_completed, "actions", env.actions)
    print("colors", color_counts(env.frame()))
    print("blobs", compact_blobs(env.frame()))
    for action, delta in action_deltas(env, env.actions).items():
        print("action", action, "delta", delta)
    points = {
        "top9": (21, 3),
        "top14": (28, 3),
        "top11": (35, 3),
        "top15": (42, 3),
        "mid1": (23, 29),
        "mid2": (29, 29),
        "mid3": (35, 29),
        "mid4": (41, 29),
        "bottom14": (19, 58),
        "bottom15": (27, 58),
        "bottom9": (35, 58),
        "bottom11": (43, 58),
        "hole_nw": (18, 25),
        "hole_se": (45, 34),
    }
    base = env.frame()
    for name, (x, y) in points.items():
        clone = env.clone()
        clone.step(6, x, y)
        print(
            "click",
            name,
            (x, y),
            "level",
            clone.levels_completed,
            "delta",
            frame_delta(base, clone.frame()),
        )
    for palette in ("bottom14", "bottom15", "bottom9", "bottom11"):
        for middle in ("mid1", "mid2", "mid3", "mid4"):
            clone = env.clone()
            px, py = points[palette]
            mx, my = points[middle]
            clone.step(6, px, py)
            clone.step(6, mx, my)
            delta = frame_delta(base, clone.frame())
            local = [
                s for s in delta["samples"]
                if 20 <= s[0] <= 40
            ]
            print(
                "pair",
                palette,
                middle,
                "level",
                clone.levels_completed,
                "local",
                local,
            )
    target_path = [
        points["bottom9"], points["mid1"],
        points["bottom14"], points["mid2"],
        points["bottom11"], points["mid3"],
        points["bottom15"], points["mid4"],
    ]
    clone = env.clone()
    for index, (x, y) in enumerate(target_path, 1):
        clone.step(6, x, y)
        middle_colors = [
            int(clone.frame()[29][c]) for c in (22, 28, 34, 40)
        ]
        print("progress", index, middle_colors, clone.levels_completed)
    for action in (5, 5):
        if not clone.terminal():
            clone.step(action)
        print("submit", action, clone.levels_completed, clone.terminal())


result = A.run_program("sb26", observe)
print("probe_result", result[0], result[2])
