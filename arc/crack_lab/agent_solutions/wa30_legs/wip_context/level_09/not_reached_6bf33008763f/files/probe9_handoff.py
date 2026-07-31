"""Observe the actual solve-loop entry state for play_level_9."""

import gkm_try

from probe9_verify import boxes, target_state


def inspect(env):
    called = {"value": False}

    def play_level_9(level_env):
        called["value"] = True
        print(
            "REAL_HANDOFF",
            {
                "terminal": level_env.terminal(),
                "avatar": boxes(level_env.frame(), 14),
                "courier_12": boxes(level_env.frame(), 12),
                "courier_15": boxes(level_env.frame(), 15),
                "target": target_state(level_env.frame()),
            },
            flush=True,
        )

    gkm_try.m.players.play_level_9 = play_level_9
    gkm_try.resumed_solve(env)
    print(
        "REAL_HANDOFF_AFTER",
        {
            "called": called["value"],
            "level": env.levels_completed,
            "terminal": env.terminal(),
        },
        flush=True,
    )


gkm_try.A.run_program("wa30", inspect)
