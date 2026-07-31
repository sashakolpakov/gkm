import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import players
from perception import connected_components


def symbolic(frame, rows, cols):
    data = np.asarray(frame)[rows, cols]
    return tuple(
        "".join("." if int(value) == 5 else f"{int(value):X}" for value in row)
        for row in data
    )


def avatar(frame):
    blobs = [
        blob
        for blob in connected_components(frame, colors=(9,), min_area=15)
        if blob.area == 15 and blob.bbox[0] < 50
    ]
    return blobs[0].bbox[:2] if len(blobs) == 1 else None


def features(frame):
    return [
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, min_area=1)
        if blob.bbox[0] < 50
        and blob.color not in (3, 4, 5)
        and blob.area >= 3
    ]


def meter(frame):
    return int(np.count_nonzero(np.asarray(frame)[61:63] == 11) // 4)


class RewardObserver:
    def __init__(self, env):
        self.env = env
        self.recent = deque(maxlen=4)
        self.steps = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        self.steps += 1
        before_level = int(self.env.levels_completed)
        before = np.asarray(self.env.frame()).copy()
        before_meter = meter(before)
        if self.steps == 90:
            outcomes = {}
            for candidate_action in self.env.actions:
                clone = self.env.clone()
                clone.step(int(candidate_action))
                outcomes[int(candidate_action)] = (
                    int(clone.levels_completed),
                    avatar(clone.frame()),
                )
            print("reward_action_context", outcomes)
        before_hud = symbolic(before, slice(55, 61), slice(3, 9))
        self.recent.append(
            (
                int(action),
                avatar(before),
                symbolic(before, slice(55, 61), slice(3, 9)),
                features(before),
                before,
            )
        )
        self.env.step(action)
        after = np.asarray(self.env.frame())
        after_meter = meter(after)
        after_hud = symbolic(after, slice(55, 61), slice(3, 9))
        if self.env.levels_completed == before_level and after_hud != before_hud:
            destination = avatar(after)
            tile = None
            if destination is not None:
                row, col = destination
                tile = symbolic(
                    before,
                    slice(max(0, row - 2), min(50, row + 3)),
                    slice(max(0, col), min(64, col + 5)),
                )
            print(
                "hud_change",
                self.steps,
                int(action),
                avatar(before),
                destination,
                before_hud,
                after_hud,
                "destination_before",
                tile,
            )
        if self.env.levels_completed == before_level and after_meter > before_meter:
            print(
                "refill",
                self.steps,
                int(action),
                avatar(before),
                avatar(after),
                before_meter,
                after_meter,
            )
        if self.env.levels_completed > before_level:
            direction = {
                1: (-5, 0),
                2: (5, 0),
                3: (0, -5),
                4: (0, 5),
            }[int(action)]
            position = avatar(before)
            target = None
            mismatch = None
            hud = np.asarray(before)[55:61:2, 3:9:2]
            if position is not None:
                new_row = position[0] + direction[0]
                new_col = position[1] + direction[1]
                target = np.asarray(before)[
                    new_row - 1 : new_row + 2,
                    new_col + 1 : new_col + 4,
                ]
                if target.shape == hud.shape:
                    mismatch = int(np.count_nonzero(hud != target))
            print(
                "reward_meter",
                before_level + 1,
                self.steps,
                int(action),
                avatar(before),
                before_meter,
                tuple(
                    "".join("." if int(value) == 5 else f"{int(value):X}" for value in row)
                    for row in hud
                ),
                None
                if target is None
                else tuple(
                    "".join("." if int(value) == 5 else f"{int(value):X}" for value in row)
                    for row in target
                ),
                mismatch,
            )
            summary = [
                (item[0], item[1], item[2], item[3])
                for item in self.recent
            ]
            print("reward_transition", before_level + 1, summary)
            position = avatar(before)
            if position is not None:
                row, col = position
                print(
                    "contact_crop",
                    "/".join(
                        symbolic(
                            before,
                            slice(max(0, row - 8), min(50, row + 10)),
                            slice(max(0, col - 8), min(64, col + 13)),
                        )
                    ),
                )


def probe(env):
    target_level = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    for level in range(1, target_level):
        getattr(players, f"play_level_{level}")(env)
    getattr(players, f"play_level_{target_level}")(RewardObserver(env))


arena.run_program("ls20", probe)
