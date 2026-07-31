"""Report public-action counts at each solved level boundary."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import players
import legs


class CountingEnv:
    def __init__(self, env):
        self.env = env
        self.actions = env.actions
        self.count = 0
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
        self.count += 1
        result = self.env.step(*action)
        if np.array_equal(before, np.asarray(self.env.frame())):
            self.unchanged.append((self.count, action))
        return result


def measure(env):
    counted = CountingEnv(env)
    calls = []
    primitives = {}
    for name in (
        "extend_tether",
        "move_vertical_lanes",
        "retract_tether",
        "repeat_action",
        "select_live_collector",
    ):
        original = getattr(legs, name)
        primitives[name] = original

        def traced(target, *args, _name=name, _original=original):
            start = target.count + 1
            result = _original(target, *args)
            calls.append((_name, args, start, target.count))
            return result

        setattr(legs, name, traced)

    for level in range(1, 9):
        player = getattr(players, f"play_level_{level}", None)
        if player is None:
            break
        before_level = counted.levels_completed
        before_count = counted.count
        before_unchanged = len(counted.unchanged)
        player(counted)
        print(
            "BOUNDARY",
            level,
            counted.levels_completed,
            counted.count - before_count,
            counted.count,
            counted.levels_completed > before_level,
        )
        print("UNCHANGED", level, counted.unchanged[before_unchanged:])
        if level == 5:
            unchanged_steps = {
                index for index, _ in counted.unchanged[before_unchanged:]
            }
            print(
                "UNCHANGED_CALLS",
                [
                    (name, args, start, end, sorted(unchanged_steps & set(range(start, end + 1))))
                    for name, args, start, end in calls
                    if start >= before_count + 1
                    and unchanged_steps & set(range(start, end + 1))
                ],
            )
        if level == 7:
            wanted = {before_count + offset for offset in (40, 41, 42)}
            print(
                "LEVEL7_CALLS_40_42",
                [
                    (name, args, start, end, sorted(wanted & set(range(start, end + 1))))
                    for name, args, start, end in calls
                    if wanted & set(range(start, end + 1))
                ],
            )
        if level == 6:
            offsets = (17, 18, 20, 47, 48, 73, 74, 96, 97, 141, 142, 143, 144, 145, 146)
            wanted = {before_count + offset for offset in offsets}
            print(
                "LEVEL6_REDUCED_CALLS",
                [
                    (name, args, start, end, sorted(wanted & set(range(start, end + 1))))
                    for name, args, start, end in calls
                    if wanted & set(range(start, end + 1))
                ],
            )
        if counted.levels_completed == before_level:
            break


levels, path, err = arena.run_program("sk48", measure)
print("MEASURE_RESULT", levels, len(path), err)
