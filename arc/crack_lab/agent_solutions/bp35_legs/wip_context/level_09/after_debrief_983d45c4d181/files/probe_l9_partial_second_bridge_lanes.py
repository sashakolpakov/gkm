"""Compare partial second-chamber bridges across all post-flip lanes."""

import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import controls
from probe_l9_route_deletions import enter_level_9, replay, route
from probe_l9_twelve_fast_frontier import SKIPS


BRIDGE_XS = (57, 51, 45, 39, 33, 27, 21, 15, 9)


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def visible(env, color):
    return tuple(
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(color,), min_area=2)
        if blob.bbox[0] < 63
    )


def key_frame(env):
    frame = np.asarray(env.frame()).copy()
    frame[63, :] = 0
    return frame.tobytes()


def probe(env):
    enter_level_9(env)
    chamber = replay(env, route(), skips=SKIPS)
    for bridge_len in range(10):
        staged = chamber.clone()
        for x in BRIDGE_XS[:bridge_len]:
            staged.step(6, x, 45)
        staged.step(*controls(staged)[0])
        row3 = "".join(
            "*" if int(staged.frame()[21][3 + 6 * col]) in (12, 15) else "."
            for col in range(10)
        )
        row5 = "".join(
            "*" if int(staged.frame()[33][3 + 6 * col]) in (12, 15) else "."
            for col in range(10)
        )
        print("BRIDGE", bridge_len, "rows", row3, row5, flush=True)
        for lane in range(10):
            child = staged.clone()
            for _ in range(lane):
                child.step(4)
                if child.terminal():
                    break
            changed = 0
            safe = lane
            last = key_frame(child)
            while not child.terminal() and int(child.levels_completed) < 9:
                avatars = visible(child, 9)
                if not avatars:
                    break
                _, left, _, right = avatars[0][0]
                col = round(((left + right) / 2 - 3) / 6)
                child.step(6, 3 + 6 * col, 33)
                safe += 1
                now = key_frame(child)
                if now != last:
                    changed += 1
                last = now
            goals = visible(child, 7)
            yellows = visible(child, 14)
            if (
                int(child.levels_completed) >= 9
                or goals
                or changed >= 4
                or (not child.terminal() and safe >= 18 - bridge_len)
            ):
                print(
                    "LANE",
                    bridge_len,
                    lane,
                    "safe",
                    safe,
                    "changed",
                    changed,
                    "win",
                    int(child.levels_completed) >= 9,
                    "terminal",
                    bool(child.terminal()),
                    "avatar",
                    visible(child, 9),
                    "goals",
                    goals,
                    "yellow",
                    yellows,
                    flush=True,
                )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
