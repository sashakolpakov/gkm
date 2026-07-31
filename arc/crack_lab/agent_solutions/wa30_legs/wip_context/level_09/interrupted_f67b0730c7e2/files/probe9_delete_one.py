"""Test one-action prefix deletions against the proven level-9 suffix."""

import gkm_try

from probe9_candidate import direct_second_prefix
from probe9_picksearch_finish import COMBINED_DISMISS_PICK
from probe9_verify import ReachedLevel9, StopAtLevel9, boxes, target_state


POSITION_BLOCK = [5, 4] + [1] * 6 + [4] * 5
PLACE_BLOCK = [4, 5, 1, 3, 2, 5, 2, 5, 1]


def replay(base, route):
    clone = base.clone()
    base_level = clone.levels_completed
    best = 0
    used = 0
    for action in route:
        if clone.terminal() or clone.levels_completed > base_level:
            break
        clone.step(action)
        used += 1
        best = max(best, len(target_state(clone.frame())["filled"]))
    return {
        "reward": clone.levels_completed - base_level,
        "terminal": clone.terminal(),
        "used": used,
        "best": best,
        "target": target_state(clone.frame()),
        "thief": boxes(clone.frame(), 15),
    }


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel9(env))
    except ReachedLevel9:
        pass

    prefix = direct_second_prefix()
    tail = COMBINED_DISMISS_PICK + POSITION_BLOCK + PLACE_BLOCK + [5]
    results = []
    for index in range(len(prefix)):
        route = prefix[:index] + prefix[index + 1:] + tail
        result = replay(env, route)
        if result["reward"] or result["best"] >= 7:
            print(
                "DELETE_ONE",
                index,
                prefix[index],
                result,
                flush=True,
            )
        results.append((result["reward"], result["best"], index, result))
    print("DELETE_BEST", max(results, key=lambda item: item[:3]), flush=True)

    replacement_results = []
    for index, original in enumerate(prefix):
        for replacement in env.actions:
            if replacement == original:
                continue
            candidate = list(prefix)
            candidate[index] = replacement
            result = replay(
                env,
                candidate
                + COMBINED_DISMISS_PICK
                + POSITION_BLOCK
                + PLACE_BLOCK,
            )
            if result["reward"] or result["best"] >= 8:
                print(
                    "REPLACE_ONE",
                    index,
                    original,
                    replacement,
                    result,
                    flush=True,
                )
            replacement_results.append(
                (result["reward"], result["best"], index, replacement, result)
            )
    print(
        "REPLACE_BEST",
        max(replacement_results, key=lambda item: item[:4]),
        flush=True,
    )

    near_route = (
        prefix
        + COMBINED_DISMISS_PICK
        + POSITION_BLOCK
        + PLACE_BLOCK
    )
    full_results = []
    for index, original in enumerate(near_route):
        result = replay(
            env,
            near_route[:index] + near_route[index + 1:] + [5],
        )
        if result["reward"] or result["best"] >= 8:
            print("FULL_DELETE", index, original, result, flush=True)
        full_results.append((result["reward"], result["best"], index, result))
    print(
        "FULL_DELETE_BEST",
        max(full_results, key=lambda item: item[:3]),
        flush=True,
    )

    full_replace_results = []
    for index, original in enumerate(near_route):
        for replacement in env.actions:
            if replacement == original:
                continue
            candidate = list(near_route)
            candidate[index] = replacement
            result = replay(env, candidate)
            if result["reward"]:
                print(
                    "FULL_REPLACE_WIN",
                    index,
                    original,
                    replacement,
                    result,
                    flush=True,
                )
                return
            full_replace_results.append(
                (result["reward"], result["best"], index, replacement, result)
            )
    print(
        "FULL_REPLACE_BEST",
        max(full_replace_results, key=lambda item: item[:4]),
        flush=True,
    )


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
