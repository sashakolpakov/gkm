"""Test row-8 courier staging points after the fast local pickup."""

import gkm_try

from probe9_prefix_shortcuts import direct_short_prefix, reach_level_9
from probe9_verify import boxes, target_state


PREFIX = (
    direct_short_prefix()
    + [3] * 5 + [5]
    + [1, 3, 3, 5]
)


def compact(env):
    return {
        "level": env.levels_completed,
        "terminal": env.terminal(),
        "avatar": boxes(env.frame(), 14),
        "helpers": boxes(env.frame(), 12),
        "cargo": boxes(env.frame(), 4),
        "target": target_state(env.frame()),
    }


def inspect(env):
    reach_level_9(env)
    picked = env.clone()
    for action in PREFIX:
        picked.step(action)
    for rights in range(0, 11):
        for ending in ([5], [5, 2]):
            child = picked.clone()
            path = [4] * rights + ending
            for action in path:
                child.step(action)
            staged = compact(child)
            turn = len(PREFIX) + len(path)
            while (
                not child.terminal()
                and child.levels_completed == env.levels_completed
            ):
                child.step(5)
                turn += 1
            final = compact(child)
            print(
                "SHORT_STAGE",
                {
                    "rights": rights,
                    "ending": ending,
                    "stage_turn": len(PREFIX) + len(path),
                    "staged": staged,
                    "end_turn": turn,
                    "final": final,
                },
                flush=True,
            )


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
