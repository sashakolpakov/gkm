"""Test the trapped 1x1 color-14 object as a second level-6 agent."""
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve


A = (6, 56, 8)
B = (6, 50, 26)
S = (6, 50, 46)


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def apply(env, path):
    for action in path:
        step(env, action)


def mini_signature(env):
    frame = np.asarray(env.frame())
    components = perception.connected_components(
        frame, colors=(14,), min_area=1
    )
    return tuple(
        (blob.bbox, blob.area)
        for blob in components
        if blob.bbox[1] < 40 and blob.area < 4
    )


def solid_exits(env):
    frame = np.asarray(env.frame())
    return tuple(
        (row, col)
        for row in range(0, 62, 2)
        for col in range(0, 40, 2)
        if np.all(frame[row:row + 2, col:col + 2] == 11)
    )


def observe(env):
    solve.solve(env)
    base_level = env.levels_completed
    print("MINI_ENTRY", mini_signature(env), flush=True)
    tested = 0
    for a_phase in range(6):
        for b_phase in range(4):
            for s_phase in range(4):
                configured = env.clone()
                apply(
                    configured,
                    [A] * a_phase + [B] * b_phase + [S] * s_phase,
                )
                before_crop = np.asarray(
                    configured.frame()
                )[54:62, 30:40].copy()
                before_mini = mini_signature(configured)
                for action in (1, 2, 3, 4):
                    child = configured.clone()
                    child.step(action)
                    after_crop = np.asarray(child.frame())[54:62, 30:40]
                    changed = np.argwhere(before_crop != after_crop)
                    after_mini = mini_signature(child)
                    tested += 1
                    if (
                        len(changed)
                        or after_mini != before_mini
                        or child.levels_completed > base_level
                        or solid_exits(child)
                    ):
                        print(
                            "MINI_EFFECT",
                            (a_phase, b_phase, s_phase),
                            "action", action,
                            "mini", before_mini, "to", after_mini,
                            "delta", int(len(changed)),
                            "exits", solid_exits(child),
                            "level", child.levels_completed,
                            flush=True,
                        )
    print("MINI_DONE", tested, flush=True)


arena.run_program("dc22", observe)
