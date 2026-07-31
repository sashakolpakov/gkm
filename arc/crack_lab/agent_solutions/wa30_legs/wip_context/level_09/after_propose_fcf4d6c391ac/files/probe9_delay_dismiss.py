"""Sweep compact thief-interception timing after the short level-9 prefix."""

import gkm_try

from probe9_prefix_shortcuts import direct_short_prefix, reach_level_9
from probe9_short_tail_search import avatar_cell
from probe9_verify import boxes, target_state


def inspect(env):
    reach_level_9(env)
    short = direct_short_prefix()
    seen = set()
    wins = []
    for delay in range(5):
        delayed = env.clone()
        for action in short + [2] * delay:
            delayed.step(action)
        candidates = []
        for waits in range(4):
            for lefts in range(1, 8):
                candidates.extend((
                    [2] * waits + [3] * lefts + [5],
                    [3] * lefts + [2] * waits + [5],
                ))
                if lefts >= 2:
                    candidates.append(
                        [3] * (lefts - 1) + [5, 3] + [2] * waits
                    )
        for path in candidates:
            key = (delay, tuple(path))
            if key in seen:
                continue
            seen.add(key)
            child = delayed.clone()
            for action in path:
                child.step(action)
            if not boxes(child.frame(), 15):
                result = (
                    delay + len(path),
                    delay,
                    path,
                    avatar_cell(child.frame()),
                    target_state(child.frame())["filled"],
                )
                wins.append(result)
    for result in sorted(wins)[:20]:
        print("DELAY_DISMISS_WIN", result, flush=True)
    print("DELAY_DISMISS_TOTAL", len(wins), flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
