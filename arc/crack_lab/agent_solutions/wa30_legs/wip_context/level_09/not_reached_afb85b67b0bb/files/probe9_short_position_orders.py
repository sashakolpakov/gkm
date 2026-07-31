"""Test one-move-shorter positioning after the verified level-9 pickup."""

import gkm_try

from perception import arr
from probe9_candidate import direct_second_prefix
from probe9_picksearch_finish import COMBINED_DISMISS_PICK
from probe9_prefix_shortcuts import reach_level_9
from probe9_verify import boxes, target_state


SUCCESSFUL_NINE = [4, 5, 1, 3, 2, 5, 2, 5, 1]


def positioned_states(base, required_up, required_right):
    frontier = {(0, 0, arr(base.frame()).tobytes()): (base.clone(), [])}
    for _ in range(required_up + required_right):
        next_states = {}
        for (used_up, used_right, _), (node, path) in frontier.items():
            choices = []
            if used_up < required_up:
                choices.append((1, used_up + 1, used_right))
            if used_right < required_right:
                choices.append((4, used_up, used_right + 1))
            for action, next_up, next_right in choices:
                child = node.clone()
                child.step(action)
                child_path = path + [action]
                key = (next_up, next_right, arr(child.frame()).tobytes())
                next_states.setdefault(key, (child, child_path))
        frontier = next_states
    return list(frontier.values())


def replay(base, route):
    child = base.clone()
    base_level = child.levels_completed
    for action in route:
        if child.terminal() or child.levels_completed > base_level:
            break
        child.step(action)
    return child


def repair_suffixes():
    yielded = set()
    candidates = [SUCCESSFUL_NINE]
    for index in range(len(SUCCESSFUL_NINE) + 1):
        for action in (1, 2, 3, 4, 5):
            candidates.append(
                SUCCESSFUL_NINE[:index]
                + [action]
                + SUCCESSFUL_NINE[index:]
            )
    for index, original in enumerate(SUCCESSFUL_NINE):
        for action in (1, 2, 3, 4, 5):
            if action != original:
                candidates.append(
                    SUCCESSFUL_NINE[:index]
                    + [action]
                    + SUCCESSFUL_NINE[index + 1:]
                )
    for candidate in candidates:
        key = tuple(candidate)
        if key not in yielded:
            yielded.add(key)
            yield candidate


def inspect(env):
    reach_level_9(env)
    picked = env.clone()
    pickup_prefix = direct_second_prefix() + COMBINED_DISMISS_PICK + [5]
    for action in pickup_prefix:
        picked.step(action)
    all_states = {}
    for up, right in ((6, 5), (5, 6)):
        for state, path in positioned_states(picked, up, right):
            all_states.setdefault(
                arr(state.frame()).tobytes(),
                (state, path, (up, right)),
            )
    print(
        "SHORT_POSITION_STATES",
        len(pickup_prefix),
        len(all_states),
        tuple(
            (
                counts,
                path,
                boxes(state.frame(), 14),
                target_state(state.frame()),
            )
            for state, path, counts in all_states.values()
        ),
        flush=True,
    )
    best = None
    for state, path, counts in all_states.values():
        for suffix in repair_suffixes():
            child = replay(state, suffix)
            reward = child.levels_completed - state.levels_completed
            target = target_state(child.frame())
            score = (reward, len(target["filled"]), -len(target["empty"]))
            if best is None or score > best[0]:
                best = (score, counts, path, suffix, target)
            if reward:
                print(
                    "SHORT_POSITION_WIN",
                    {
                        "counts": counts,
                        "position": path,
                        "suffix": suffix,
                        "target": target,
                    },
                    flush=True,
                )
                return
    print("SHORT_POSITION_BEST", best, flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
