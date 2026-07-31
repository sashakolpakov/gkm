import sys
from collections import Counter

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import perception
import players


def enter_level_4(env):
    players.play_level_1(env)
    players.play_level_2(env)
    players.play_level_3(env)


def transition_counts(before, after):
    before = np.asarray(before)
    after = np.asarray(after)
    changed = before != after
    return sorted(
        Counter(
            (int(a), int(b))
            for a, b in zip(before[changed], after[changed])
        ).items()
    )


def center_map(frame):
    frame = np.asarray(frame)
    return tuple(
        "".join(f"{int(frame[2 + 5 * r, 1 + 5 * c]):X}" for c in range(12))
        for r in range(12)
    )


def changed_points(before, after):
    before = np.asarray(before)
    after = np.asarray(after)
    ys, xs = np.where(before != after)
    return [
        (int(y), int(x), int(before[y, x]), int(after[y, x]))
        for y, x in zip(ys, xs)
        if y < 60
    ]


def probe(env):
    enter_level_4(env)
    print(
        "entry",
        "level",
        int(env.levels_completed),
        "terminal",
        bool(env.terminal()),
        "actions",
        env.actions,
    )
    print("colors", sorted(perception.color_counts(env.frame()).items()))
    print("centers", *center_map(env.frame()), sep="\n")
    print(
        "detail_components",
        [
            (b.color, b.bbox, b.area)
            for b in perception.connected_components(
                env.frame(), colors=(0, 1, 8, 9, 11, 12, 14), min_area=1
            )
            if b.bbox[0] < 60
        ],
    )
    print(
        "objects",
        [
            (o["color"], o["bbox"], o["area"])
            for o in perception.object_candidates(env.frame(), min_area=4)
            if o["area"] < 1000
        ],
    )

    before = np.asarray(env.frame()).copy()
    for action in env.actions:
        clone = env.clone()
        clone.step(int(action))
        delta = perception.frame_delta(before, clone.frame())
        print(
            "action",
            int(action),
            "level",
            int(clone.levels_completed),
            "delta",
            delta["count"],
            delta["bbox"],
            "transitions",
            transition_counts(before, clone.frame()),
            "board_points",
            changed_points(before, clone.frame()),
        )


arena.run_program("ls20", probe)
