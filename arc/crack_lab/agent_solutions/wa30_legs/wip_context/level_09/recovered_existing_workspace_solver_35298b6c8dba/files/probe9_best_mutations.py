"""Mutate the exact all-eight-at-69 level-9 post-pick route."""

import gkm_try

from probe9_candidate import direct_second_prefix
from probe9_picksearch_finish import COMBINED_DISMISS_PICK
from probe9_prefix_shortcuts import reach_level_9
from probe9_verify import target_state


POSITION = [4] + [1] * 6 + [4] * 4
SUFFIX = [2, 4, 5, 1, 3, 2, 5, 2, 5, 1]
TAIL = POSITION + SUFFIX


def replay(base, route):
    child = base.clone()
    base_level = child.levels_completed
    best = 0
    for action in route:
        if child.terminal() or child.levels_completed > base_level:
            break
        child.step(action)
        best = max(best, len(target_state(child.frame())["filled"]))
    return child, best


def inspect(env):
    reach_level_9(env)
    picked = env.clone()
    prefix = direct_second_prefix() + COMBINED_DISMISS_PICK + [5]
    for action in prefix:
        picked.step(action)
    exact, exact_best = replay(picked, TAIL)
    print(
        "BEST_MUTATION_EXACT",
        len(prefix) + len(TAIL),
        exact.levels_completed - picked.levels_completed,
        exact_best,
        target_state(exact.frame()),
        flush=True,
    )

    best = None
    for index in range(len(TAIL)):
        shortened = TAIL[:index] + TAIL[index + 1:]
        for final_action in picked.actions:
            child, filled = replay(picked, shortened + [final_action])
            reward = child.levels_completed - picked.levels_completed
            target = target_state(child.frame())
            score = (reward, filled, len(target["filled"]))
            if best is None or score > best[0]:
                best = (
                    score,
                    "delete",
                    index,
                    TAIL[index],
                    final_action,
                    target,
                )
            if reward:
                print(
                    "BEST_MUTATION_WIN",
                    shortened + [final_action],
                    best,
                    flush=True,
                )
                return

    for index, original in enumerate(TAIL):
        for replacement in picked.actions:
            if replacement == original:
                continue
            route = list(TAIL)
            route[index] = replacement
            child, filled = replay(picked, route)
            reward = child.levels_completed - picked.levels_completed
            target = target_state(child.frame())
            score = (reward, filled, len(target["filled"]))
            if best is None or score > best[0]:
                best = (
                    score,
                    "replace",
                    index,
                    original,
                    replacement,
                    target,
                )
            if reward:
                print("BEST_MUTATION_WIN", route, best, flush=True)
                return

    for index in range(len(TAIL) - 1):
        route = list(TAIL)
        route[index], route[index + 1] = route[index + 1], route[index]
        child, filled = replay(picked, route)
        reward = child.levels_completed - picked.levels_completed
        target = target_state(child.frame())
        score = (reward, filled, len(target["filled"]))
        if best is None or score > best[0]:
            best = (
                score,
                "swap",
                index,
                TAIL[index:index + 2],
                target,
            )
        if reward:
            print("BEST_MUTATION_WIN", route, best, flush=True)
            return

    print("BEST_MUTATION_BEST", best, flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
