"""Compact reproduction of the actual solve-loop entry state for level 9."""

import gkm_try

from perception import action_deltas, color_counts, connected_components
from probe9_verify import boxes, target_state, tile_map


class CaptureLevel9:
    def __init__(self, env):
        self.env = env
        self.actions_seen = []
        self.transitions = []
        self.level9 = None
        self.level9_index = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action, *args):
        before = self.env.levels_completed
        result = self.env.step(action, *args)
        self.actions_seen.append(action)
        if self.env.levels_completed > before:
            self.transitions.append(
                (self.env.levels_completed, len(self.actions_seen))
            )
        if before < 8 <= self.env.levels_completed and self.level9 is None:
            self.level9 = self.env.clone()
            self.level9_index = len(self.actions_seen)
        return result


def compact(env):
    frame = env.frame()
    return {
        "level": env.levels_completed,
        "terminal": env.terminal(),
        "actions": tuple(env.actions),
        "colors": color_counts(frame),
        "avatar": boxes(frame, 14),
        "cargo": boxes(frame, 4),
        "courier_12": boxes(frame, 12),
        "courier_15": boxes(frame, 15),
        "target": target_state(frame),
    }


def inspect(env):
    capture = CaptureLevel9(env)
    gkm_try.resumed_solve(capture)
    print(
        "TRANSITION",
        compact(capture.level9),
        "POST_ACTIONS",
        tuple(capture.actions_seen[capture.level9_index:]),
        flush=True,
    )
    print("ALL_TRANSITIONS", tuple(capture.transitions), flush=True)
    print("ENTRY", compact(env), flush=True)
    print("MAP", *tile_map(env.frame()), sep="\n", flush=True)
    print(
        "BLOBS",
        tuple(
            (blob.color, blob.bbox, blob.area)
            for blob in connected_components(env.frame(), min_area=4)
            if blob.color not in (1, 2)
        ),
        flush=True,
    )
    after = {}
    for action, delta in action_deltas(env).items():
        child = env.clone()
        child.step(action)
        after[action] = {
            "pixels": delta["count"],
            "bbox": delta["bbox"],
            "state": compact(child),
        }
    print(
        "ACTION_DELTAS",
        after,
        flush=True,
    )


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
