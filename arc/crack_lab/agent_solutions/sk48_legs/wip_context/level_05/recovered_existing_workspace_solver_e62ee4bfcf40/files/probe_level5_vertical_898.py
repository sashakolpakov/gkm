"""Try the existing vertical weave on the verified level-5 8/9/8 stack."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import players
from search_level5_threaded import (
    CONTROLLED_PAIR,
    LOWER_FINAL_EIGHT,
    PREFIX,
    apply,
    clear_target_geometry,
    observe,
    target_prefix,
)


def summary(env):
    observation = observe(env)
    return (
        env.levels_completed,
        observation[2],
        observation[3],
        observation[4],
        clear_target_geometry(observation[1]),
        observation[1],
    )


def run(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)
    apply(env, PREFIX + CONTROLLED_PAIR + LOWER_FINAL_EIGHT + (3, 3))
    print("ROOT", summary(env))

    for approach in range(7):
        for thread_steps in range(7):
            branch = env.clone()
            try:
                players.weave_vertical_four_train(
                    branch,
                    approach_lanes=approach,
                    thread_steps=thread_steps,
                )
            except IndexError:
                continue
            observation = observe(branch)
            prefix = target_prefix(observation[3])
            geometry = clear_target_geometry(observation[1])
            if (
                branch.levels_completed > env.levels_completed
                or prefix
                or geometry >= 2
            ):
                print(
                    "WEAVE",
                    approach,
                    thread_steps,
                    summary(branch),
                )


levels, path, error = arena.run_program("sk48", run)
print("END", levels, len(path), error)
