import sys
import time

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

import legs
import players


def reach_level_five(env):
    for level in range(1, 5):
        getattr(players, f"play_level_{level}")(env)
        if env.levels_completed != level:
            return False
    return True


def probe(env):
    if not reach_level_five(env):
        print("advance_failed", env.levels_completed)
        return
    root = env.clone()
    started = time.monotonic()
    path = legs.gravity_room_search(root, max_states=1200, max_depth=64)
    elapsed = round(time.monotonic() - started, 2)
    result = root.clone()
    legs.run_actions(result, path)
    print(
        "search",
        {"seconds": elapsed, "length": len(path), "path": path},
        "result",
        {
            "level": result.levels_completed,
            "terminal": result.terminal(),
            "avatar_col": legs.avatar_column(result.frame()),
        },
    )


if __name__ == "__main__":
    A.run_program("bp35", probe)
