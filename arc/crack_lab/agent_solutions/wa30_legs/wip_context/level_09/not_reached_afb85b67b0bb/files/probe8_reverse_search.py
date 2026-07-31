"""Targeted search for a faster second level-8 competitor dismissal."""

import gkm_try

from probe8_entry import ReachedLevel8, StopAtLevel8
from probe8_trace import target_state
from probe9_verify import boxes


TOP_DISMISS = [4] * 3 + [1] * 3 + [5] * 3


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel8(env))
    except ReachedLevel8:
        pass

    state = env.clone()
    for action in TOP_DISMISS:
        state.step(action)
    print(
        "SEARCH_START",
        {
            "turn": len(TOP_DISMISS),
            "avatar": boxes(state.frame(), 14),
            "competitors": boxes(state.frame(), 15),
            "target": target_state(state.frame()),
        },
        flush=True,
    )
    corridor = [2] * 3 + [4] * 5
    candidates = set()
    for down in range(2, 8):
        for left in range(0, 10):
            for up in range(0, 4):
                for right in range(0, 3):
                    for uses in range(1, 4):
                        candidates.add(
                            tuple(
                                [2] * down
                                + [3] * left
                                + [1] * up
                                + [4] * right
                                + [5] * uses
                            )
                        )
            for down_again in range(0, 4):
                for right in range(0, 3):
                    for uses in range(1, 3):
                        candidates.add(
                            tuple(
                                [2] * down
                                + [3] * left
                                + [2] * down_again
                                + [4] * right
                                + [5] * uses
                            )
                        )
    path = None
    tested = 0
    for suffix in sorted(candidates, key=lambda item: (len(item), item)):
        if len(suffix) > 11:
            break
        child = state.clone()
        for action in corridor + list(suffix):
            child.step(action)
        tested += 1
        if not boxes(child.frame(), 15):
            path = corridor + list(suffix)
            break
    print("SEARCH_RESULT", {"tested": tested, "path": path}, flush=True)
    if path is None:
        return
    for action in path:
        state.step(action)
    print(
        "SEARCH_END",
        {
            "turn": len(TOP_DISMISS) + len(path),
            "avatar": boxes(state.frame(), 14),
            "competitors": boxes(state.frame(), 15),
            "target": target_state(state.frame()),
        },
        flush=True,
    )


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
