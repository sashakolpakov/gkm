"""Check level-5 full-grid submissions and symbolic failure feedback."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import frame_delta
import players

SLOTS = [(20, 23), (26, 23), (32, 23), (38, 23), (44, 23),
         (26, 37), (32, 37), (38, 37)]
SWATCH = {15: (6, 58), 6: (14, 58), 8: (20, 58),
          11: (34, 58), 14: (42, 58), 9: (48, 58)}


def try_assignment(env, name, colors):
    clone = env.clone()
    for slot, color in zip(SLOTS, colors):
        clone.step(6, *SWATCH[color])
        clone.step(6, *slot)
    before = clone.frame().copy()
    clone.step(5)
    delta = frame_delta(before, clone.frame())
    central = [
        sample for sample in delta["samples"]
        if 15 <= sample[0] <= 42 and 13 <= sample[1] <= 50
    ]
    print(
        name,
        tuple(colors),
        "level",
        clone.levels_completed,
        "delta",
        (delta["count"], delta["bbox"]),
        "central",
        central,
    )


def probe(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)
    try_assignment(env, "palette", (15, 6, 8, 8, 11, 14, 9, 9))
    try_assignment(env, "top_first", (6, 14, 8, 8, 14, 8, 8, 11))
    try_assignment(env, "top_last", (14, 8, 8, 14, 8, 8, 11, 15))
    try_assignment(env, "split_skip", (6, 14, 8, 8, 14, 8, 11, 15))


levels, path, err = A.run_program("sb26", probe)
print("done", levels, len(path), err)
