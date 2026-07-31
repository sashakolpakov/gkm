"""Mutate the third level-8 transfer to improve the final courier handoff."""

import gkm_try

from probe8_entry import ReachedLevel8, StopAtLevel8


PREFIX = (
    [4] * 8 + [2] * 5 + [3] * 3 + [5]
    + [4] * 3 + [1] * 5 + [3] * 4 + [1] * 4
    + [5] * 2 + [3, 5]
    + [4] * 7 + [5, 2]
    + [3] * 9 + [1, 5] + [4] * 11 + [1, 5, 2]
)
THIRD = [1] + [3] * 10 + [1, 5, 2] + [4] * 11 + [1, 5, 2]


def finish(start, route, limit=120):
    clone = start.clone()
    base_level = clone.levels_completed
    turn = len(PREFIX)
    for action in route:
        clone.step(action)
        turn += 1
        if clone.levels_completed > base_level or clone.terminal():
            return clone.levels_completed > base_level, turn
    while turn < limit and clone.levels_completed == base_level:
        clone.step(5)
        turn += 1
        if clone.terminal():
            break
    return clone.levels_completed > base_level, turn


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel8(env))
    except ReachedLevel8:
        pass
    start = env.clone()
    for action in PREFIX:
        start.step(action)
    print("TAIL_MUTATION_BASE", len(PREFIX), finish(start, THIRD), flush=True)
    best = (114, None)
    for index in range(len(THIRD)):
        variants = (
            THIRD[:index] + THIRD[index + 1:],
            *(
                THIRD[:index] + [action] + THIRD[index + 1:]
                for action in (1, 2, 3, 4, 5)
                if action != THIRD[index]
            ),
        )
        for variant in variants:
            won, turn = finish(start, variant)
            if won and turn < best[0]:
                best = (turn, (index, variant[index:index + 1], variant))
                print("TAIL_MUTATION_BEST", best, flush=True)
    for index in range(len(THIRD) + 1):
        for action in (1, 2, 3, 4, 5):
            variant = THIRD[:index] + [action] + THIRD[index:]
            won, turn = finish(start, variant)
            if won and turn < best[0]:
                best = (turn, ("insert", index, action, variant))
                print("TAIL_MUTATION_BEST", best, flush=True)
    print("TAIL_MUTATION_RESULT", best, flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
