"""Enumerate compact drop endings onto the two-stage courier path."""

from itertools import product

import gkm_try

from perception import arr
from probe9_two_staged_trace import DISMISS, TWO_STAGED
from probe9_verify import boxes, target_state


STAGE_BASE = [3] * 2 + [5, 4] + [1] * 5


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
    gkm_try.resumed_solve(env)
    base = env.clone()
    prefix = TWO_STAGED + DISMISS + STAGE_BASE
    for action in prefix:
        base.step(action)
    candidates = {}
    for length in range(1, 4):
        for ending in product(base.actions, repeat=length):
            staged = base.clone()
            for action in ending:
                staged.step(action)
            candidates.setdefault(
                arr(staged.frame()).tobytes(),
                (staged, list(ending)),
            )
    best = None
    for staged, ending in candidates.values():
        child = staged.clone()
        while (
            not child.terminal()
            and child.levels_completed == env.levels_completed
        ):
            child.step(5)
        target = target_state(child.frame())
        score = (
            child.levels_completed - env.levels_completed,
            len(target["filled"]),
            -len(target["empty"]),
        )
        result = (
            score,
            ending,
            13 + len(prefix) + len(ending),
            compact(staged),
            compact(child),
        )
        if best is None or score > best[0]:
            best = result
        if score[0]:
            print("STAGE_ENDING_WIN", result, flush=True)
            return
    print("STAGE_ENDING_STATES", len(candidates), flush=True)
    print("STAGE_ENDING_BEST", best, flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
