"""Verify whether a successful capture leaves its destination selected."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from perception import arr, frame_delta, safe_step


REFERENCE = (
    (6, 18, 19), (6, 30, 19),
    (6, 30, 19), (6, 42, 19),
    (6, 42, 19), (6, 42, 31),
    (6, 42, 31), (6, 42, 43),
)

CHAINED = (
    (6, 18, 19), (6, 30, 19),
    (6, 42, 19), (6, 42, 31), (6, 42, 43),
)


def probe(env):
    reference = env.clone()
    for action in REFERENCE:
        safe_step(reference, action)
    chained = env.clone()
    for action in CHAINED:
        safe_step(chained, action)
    before = arr(reference.frame()).copy()
    after = arr(chained.frame()).copy()
    after[0] = before[0]
    print(
        "capture_chain",
        int(reference.levels_completed),
        int(chained.levels_completed),
        frame_delta(before, after),
        flush=True,
    )


arena.run_program("lf52", probe)
