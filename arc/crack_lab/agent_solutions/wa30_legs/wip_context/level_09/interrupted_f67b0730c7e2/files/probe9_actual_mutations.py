"""One-edit search around the verified seven-of-eight actual-entry route."""

import gkm_try

from probe9_two_staged_trace import DISMISS, LOCAL_FINISH, TWO_STAGED
from probe9_verify import boxes, target_state


BASE_ROUTE = TWO_STAGED + DISMISS + LOCAL_FINISH + [5, 5]


def evaluate(base, route):
    child = base.clone()
    base_level = child.levels_completed
    best_filled = 0
    best_turn = 0
    for turn, action in enumerate(route, 14):
        if child.terminal() or child.levels_completed > base_level:
            break
        child.step(action)
        filled = len(target_state(child.frame())["filled"])
        if filled > best_filled:
            best_filled = filled
            best_turn = turn
    target = target_state(child.frame())
    score = (
        child.levels_completed - base_level,
        best_filled,
        len(target["filled"]),
        -len(target["empty"]),
        -best_turn,
    )
    return score, child, target


def inspect(env):
    gkm_try.resumed_solve(env)
    base = env.clone()
    candidates = [("base", None, None, BASE_ROUTE)]
    for index, original in enumerate(BASE_ROUTE):
        for replacement in base.actions:
            if replacement != original:
                route = list(BASE_ROUTE)
                route[index] = replacement
                candidates.append(("replace", index, replacement, route))
        candidates.append(
            (
                "delete",
                index,
                original,
                BASE_ROUTE[:index] + BASE_ROUTE[index + 1:] + [5],
            )
        )
    for index in range(len(BASE_ROUTE) - 1):
        if BASE_ROUTE[index] != BASE_ROUTE[index + 1]:
            route = list(BASE_ROUTE)
            route[index], route[index + 1] = route[index + 1], route[index]
            candidates.append(("swap", index, None, route))

    ranked = []
    for kind, index, value, route in candidates:
        score, child, target = evaluate(base, route)
        result = (
            score,
            kind,
            index,
            value,
            route,
            target,
            boxes(child.frame(), 0),
            boxes(child.frame(), 12),
        )
        ranked.append(result)
        if score[0]:
            print("ACTUAL_MUTATION_WIN", result, flush=True)
            return
    ranked.sort(key=lambda item: item[0], reverse=True)
    print("ACTUAL_MUTATION_COUNT", len(ranked), flush=True)
    for result in ranked[:20]:
        print("ACTUAL_MUTATION_BEST", result, flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
