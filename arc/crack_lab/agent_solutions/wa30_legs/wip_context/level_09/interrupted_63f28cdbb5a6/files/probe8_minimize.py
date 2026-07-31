"""Greedily delete actions from the proven level-8 route on bounded clones."""

import gkm_try

from probe8_entry import ReachedLevel8, StopAtLevel8
from probe8_trace import target_state


def active_route():
    return (
        [4] * 8 + [2] * 5 + [3] * 3 + [5]
        + [4] * 3 + [1] * 5 + [3] * 5 + [1] * 4
        + [3, 1] + [5] * 3
        + [3, 5, 1] + [4] * 8 + [5, 2]
        + [3] * 10 + [1, 5, 2] + [4] * 11 + [1, 5, 2]
    )


def evaluate(base, actions, max_turns=150):
    clone = base.clone()
    base_level = clone.levels_completed
    used = 0
    for action in actions:
        if clone.terminal() or clone.levels_completed > base_level:
            break
        clone.step(action)
        used += 1
    while (
        used < max_turns
        and not clone.terminal()
        and clone.levels_completed == base_level
    ):
        clone.step(5)
        used += 1
    empty, filled = target_state(clone.frame())
    return {
        "won": clone.levels_completed > base_level,
        "turns": used,
        "empty": empty,
        "filled": filled,
    }


def minimize(base):
    best = active_route()
    best_result = evaluate(base, best)
    print("MIN_START", len(best), best_result, flush=True)
    for chunk in (16, 8, 4, 2, 1):
        changed = True
        while changed:
            changed = False
            index = 0
            while index < len(best):
                candidate = best[:index] + best[index + chunk:]
                result = evaluate(base, candidate)
                if result["won"] and result["turns"] < best_result["turns"]:
                    print(
                        "MIN_ACCEPT",
                        chunk,
                        index,
                        len(candidate),
                        result,
                        flush=True,
                    )
                    best = candidate
                    best_result = result
                    changed = True
                else:
                    index += chunk
    return best, best_result


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel8(env))
    except ReachedLevel8:
        pass
    best, result = minimize(env)
    print("MIN_RESULT", best, result, flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
