"""Verify unchanged-frame prefix actions by deletion on pristine clones."""

import sys
from itertools import combinations

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import players


class RecordingEnv:
    def __init__(self, env):
        self.env = env
        self.actions = env.actions
        self.path = []
        self.unchanged = []

    @property
    def levels_completed(self):
        return self.env.levels_completed

    def frame(self):
        return self.env.frame()

    def terminal(self):
        return self.env.terminal()

    def step(self, *action):
        before = np.asarray(self.env.frame()).copy()
        self.path.append(action)
        result = self.env.step(*action)
        if np.array_equal(before, np.asarray(self.env.frame())):
            self.unchanged.append(len(self.path) - 1)
        return result


def replay(root, path, omitted):
    clone = root.clone()
    for index, action in enumerate(path):
        if index in omitted:
            continue
        clone.step(*action)
    return clone


def probe(env):
    for level in range(1, 6):
        getattr(players, f"play_level_{level}")(env)
    root = env.clone()
    recorded = RecordingEnv(root.clone())
    players.play_level_6(recorded)
    baseline = np.asarray(recorded.frame()).copy()
    print("LEVEL6_PATH", len(recorded.path), flush=True)

    groups = (
        tuple(range(16, 20)),
        tuple(range(46, 49)),
        tuple(range(72, 75)),
        tuple(range(95, 101)),
        tuple(range(140, 146)),
    )
    chosen = []
    for group in groups:
        best = ()
        for size in range(len(group), 0, -1):
            for subset in combinations(group, size):
                clone = replay(root, recorded.path, set(subset))
                if (
                    clone.levels_completed == 6
                    and np.array_equal(np.asarray(clone.frame()), baseline)
                ):
                    best = subset
                    break
            if best:
                break
        chosen.extend(best)
        print(
            "GROUP_BEST",
            [index + 1 for index in group],
            [index + 1 for index in best],
            flush=True,
        )

    combined = replay(root, recorded.path, set(chosen))
    print(
        "GROUPS_COMBINED",
        combined.levels_completed == 6
        and np.array_equal(np.asarray(combined.frame()), baseline),
        [index + 1 for index in chosen],
        len(recorded.path) - len(chosen),
        flush=True,
    )



levels, path, err = arena.run_program("sk48", probe)
print("DELETE_RESULT", levels, len(path), err)
