"""Validate the lower-lane 8-9-8 assembly from the vertical frontier."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import players
from search_level5_threaded import (
    CONTROLLED_PAIR,
    LOWER_FINAL_EIGHT,
    PREFIX,
    apply,
    observe,
)


MAKE_TWO_EIGHTS = (3, 3, 2, 2, 4, 4)
CLEAN_PREFIX = tuple(action for action in PREFIX if action != 6)
CLEAN_LOWER = tuple(
    action
    for action, count in (
        (1, 2), (2, 2), (4, 5), (1, 2), (2, 1), (3, 2),
        (2, 1), (1, 2), (2, 1), (4, 3), (1, 2), (4, 3), (2, 2),
    )
    for _ in range(count)
)


def summary(env):
    observation = observe(env)
    return (
        env.levels_completed,
        env.terminal(),
        observation[2],
        observation[3],
        observation[4],
        observation[1],
    )


def run(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)
    apply(
        env,
        CLEAN_PREFIX
        + CONTROLLED_PAIR
        + CLEAN_LOWER
        + (3, 3)
        + MAKE_TWO_EIGHTS,
    )
    print("TWO_EIGHTS", summary(env))
    for label, path in (
        ("LOWER_FIRST", (2,)),
        ("DETACH_FIRST", (3,) * 5),
        ("PARK_FIRST", (4,) * 4 + (3,) * 4),
        ("APPROACH_SECOND", (1,)),
        ("PARK_SECOND", (4,) * 6 + (3,) * 6),
        ("REACH_TOP_NINE", (1,) * 4),
        ("THREAD_TOP_NINE", (4,) * 7),
        ("PULL_NINE_LEFT", (3,) * 4),
        ("CARRY_NINE_DOWN", (2,) * 4),
        ("DETACH_NINE", (3,) * 3),
        ("PARK_NINE", (4,) * 5 + (3,) * 5),
        ("APPROACH_FIRST", (2,) * 2 + (4,) * 5),
        ("PUSH_FIRST_UP", (1,)),
        ("RESET_AFTER_PUSH", (3,)),
        ("ENTER_TARGET_ROW", (1,)),
        ("THREAD_TARGET_1", (4,)),
        ("THREAD_TARGET_2", (4,)),
        ("THREAD_TARGET_3", (4,)),
    ):
        try:
            apply(env, path)
        except IndexError:
            print("INVALID", label, "level", env.levels_completed)
            return
        print(label, summary(env))
        if label == "PUSH_FIRST_UP":
            paths = [
                (1,), (2,), (3,), (4,), (6,),
                (2, 3), (2, 4), (2, 6, 3), (2, 6, 4),
                (2, 2, 3), (2, 2, 4), (2, 1, 3), (2, 1, 4),
            ]
            paths.extend((3,) * count for count in range(2, 7))
            paths.extend((2,) + (3,) * count for count in range(2, 7))
            reset = (3,) * 4
            paths.extend(
                reset + suffix
                for suffix in (
                    (1,), (2,), (4,), (6,),
                    (2, 1), (2, 1, 1), (2, 4, 1),
                    (4, 1), (4, 4, 1), (2, 4, 4, 1, 1),
                )
            )
            for probe_path in paths:
                branch = env.clone()
                try:
                    apply(branch, probe_path)
                    print("POST_PUSH_PROBE", probe_path, summary(branch))
                except IndexError:
                    print("POST_PUSH_INVALID", probe_path)
        if label == "THREAD_TARGET_2":
            for finish_path in (
                (1,), (2,), (3,), (6,),
                (4,), (6, 4), (6, 6, 4), (3, 4), (3, 6, 4),
                (1, 4), (2, 4), (1, 2, 4), (2, 1, 4),
                (3, 3, 4, 4), (1, 3, 2, 4),
            ):
                branch = env.clone()
                try:
                    apply(branch, finish_path)
                    print("FINISH_PROBE", finish_path, summary(branch))
                except IndexError:
                    print(
                        "FINISH_INVALID",
                        finish_path,
                        "level", branch.levels_completed,
                    )
        if env.levels_completed > 4 or env.terminal():
            return


levels, path, error = arena.run_program("sk48", run)
print("END", levels, len(path), error)
