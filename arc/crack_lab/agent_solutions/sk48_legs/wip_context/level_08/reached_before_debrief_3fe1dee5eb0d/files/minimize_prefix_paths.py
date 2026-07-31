"""Minimize long prefix levels while preserving their exact exit frames."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import players


class Recorder:
    def __init__(self, env):
        self.env = env
        self.actions = env.actions
        self.path = []

    @property
    def levels_completed(self):
        return self.env.levels_completed

    def frame(self):
        return self.env.frame()

    def terminal(self):
        return self.env.terminal()

    def step(self, *action):
        self.path.append(action)
        return self.env.step(*action)


def replay_matches(root, path, target_level, target_frame):
    node = root.clone()
    for action in path:
        node.step(*action)
    return (
        node.levels_completed == target_level
        and np.array_equal(np.asarray(node.frame()), target_frame)
    )


def minimize(root, path, target_level, target_frame):
    if not replay_matches(root, path, target_level, target_frame):
        raise RuntimeError("recorded path does not reproduce its exit")
    pass_number = 0
    while True:
        pass_number += 1
        changed = False
        for size in (32, 16, 8, 4, 2, 1):
            index = 0
            while index + size <= len(path):
                candidate = path[:index] + path[index + size :]
                if replay_matches(root, candidate, target_level, target_frame):
                    path = candidate
                    changed = True
                    print(
                        "DROP",
                        target_level,
                        pass_number,
                        size,
                        index,
                        "LEN",
                        len(path),
                        flush=True,
                    )
                else:
                    index += size
        if not changed:
            return path


def probe(env):
    records = {}
    for level in range(1, 8):
        root = env.clone()
        recorder = Recorder(env)
        getattr(players, f"play_level_{level}")(recorder)
        records[level] = (
            root,
            recorder.path,
            env.levels_completed,
            np.asarray(env.frame()).copy(),
        )
        print(
            "RECORDED",
            level,
            len(recorder.path),
            env.levels_completed,
            flush=True,
        )

    for level in (5, 6, 7):
        root, path, target_level, target_frame = records[level]
        minimized = minimize(
            root,
            path,
            target_level,
            target_frame,
        )
        print(
            "MINIMIZED_LEVEL",
            level,
            len(minimized),
            json.dumps(minimized),
            flush=True,
        )


levels, path, err = arena.run_program("sk48", probe)
print("PREFIX_MINIMIZE_RESULT", levels, len(path), err)
