"""Compare second-cargo courier ports from the real level-9 handoff."""

from itertools import product

import gkm_try

from perception import arr
from probe9_actual_candidates import replay, state
from probe9_verify import boxes


def first_dismissal(start, max_depth=6, max_transitions=5000):
    frontier = [(start.clone(), [])]
    seen = {arr(start.frame()).tobytes()}
    transitions = 0
    for depth in range(1, max_depth + 1):
        next_frontier = []
        for node, path in frontier:
            for action in node.actions:
                child = node.clone()
                child.step(action)
                transitions += 1
                child_path = path + [action]
                if not boxes(child.frame(), 15):
                    return child_path, child, transitions
                key = arr(child.frame()).tobytes()
                if key not in seen:
                    seen.add(key)
                    next_frontier.append((child, child_path))
                if transitions >= max_transitions:
                    return None, None, transitions
        frontier = next_frontier
    return None, None, transitions


def short_finish(start, max_depth=4):
    base_level = start.levels_completed
    frontier = [(start.clone(), [])]
    for depth in range(1, max_depth + 1):
        next_frontier = []
        for node, path in frontier:
            for action in node.actions:
                child = node.clone()
                child.step(action)
                child_path = path + [action]
                if child.levels_completed > base_level:
                    return child_path, state(child)
                if not child.terminal():
                    next_frontier.append((child, child_path))
        frontier = next_frontier
    return None, None


def dismissal_variants(start, depth=3):
    variants = []
    for path in product(start.actions, repeat=depth):
        child = replay(start, path)
        if not boxes(child.frame(), 15):
            variants.append((list(path), child))
    return variants


def combined_stage_dismiss(start, max_depth=4):
    for depth in range(1, max_depth + 1):
        wins = []
        for path in product(start.actions, repeat=depth):
            child = replay(start, path)
            staged_cargo = tuple(
                bbox
                for bbox in boxes(child.frame(), 4)
                if bbox[0] == 28 and 16 <= bbox[1] <= 32
            )
            if not boxes(child.frame(), 15) and staged_cargo:
                wins.append((list(path), state(child)))
        if wins:
            return wins
    return []


def inspect(env):
    gkm_try.resumed_solve(env)
    first_delivery = [3, 1, 5] + [3] * 6 + [1] * 3 + [3, 5]
    second_pick = [2] * 2 + [4] * 5 + [1, 5]
    picked = first_delivery + second_pick

    stage_first = (
        [3, 1, 5] + [3] * 6 + [1, 5]
    )
    pick_second_after_stage = [4] * 4 + [1, 5]
    stage_second_after_stage = [2] + [3] * 6 + [1, 5]
    two_staged = (
        stage_first + pick_second_after_stage + stage_second_after_stage
    )
    two_staged_state = replay(env, two_staged)
    two_path, two_dismissed, two_transitions = first_dismissal(
        two_staged_state, max_depth=8, max_transitions=5000
    )
    print(
        "TWO_STAGED",
        {
            "turn": 13 + len(two_staged),
            "state": state(two_staged_state),
            "dismiss": two_path,
            "dismiss_turn": (
                None if two_path is None
                else 13 + len(two_staged) + len(two_path)
            ),
            "dismiss_state": (
                None if two_dismissed is None else state(two_dismissed)
            ),
            "transitions": two_transitions,
        },
        flush=True,
    )
    two_dismiss = [1, 3, 3, 3, 5]
    local_finish = (
        [3] * 2 + [5, 4]
        + [1] * 6 + [4] * 5
        + [5, 1, 3, 2, 5, 2, 5, 1]
    )
    two_candidate = two_staged + two_dismiss + local_finish
    clone = replay(env, two_candidate)
    print(
        "TWO_STAGED_FINISH",
        len(two_candidate),
        state(clone),
        short_finish(clone),
        flush=True,
    )
    for column in range(4, 9):
        stage = [2] + [3] * (12 - column) + [1, 5]
        staged = replay(env, picked + stage)
        path, dismissed, transitions = first_dismissal(staged)
        idle_state = None
        if dismissed is not None:
            idle = dismissed.clone()
            while not idle.terminal():
                idle.step(5)
            idle_state = state(idle)
        print(
            "PORT",
            {
                "column": column,
                "stage_turn": 13 + len(picked + stage),
                "stage": state(staged),
                "dismiss": path,
                "dismiss_turn": (
                    None if path is None
                    else 13 + len(picked + stage) + len(path)
                ),
                "dismiss_state": None if dismissed is None else state(dismissed),
                "idle_state": idle_state,
                "transitions": transitions,
            },
            flush=True,
        )

    stage6 = [2] + [3] * 6 + [1, 5]
    dismiss6 = [3, 1, 5]
    fill_bottom_middle = (
        [3] * 4 + [5]
        + [2] + [4] * 5 + [1] * 4 + [5, 4]
    )
    candidate = picked + stage6 + dismiss6 + fill_bottom_middle
    clone = replay(env, candidate)
    print("PORT6_MANUAL", len(candidate), state(clone), flush=True)
    prior = None
    for turn in range(13 + len(candidate), 71):
        current = state(clone)
        condensed = (
            current["empty"],
            current["filled"],
            current["helper"],
            current["level"],
            current["terminal"],
        )
        if condensed != prior:
            print("PORT6_EVENT", turn, current, flush=True)
        if clone.terminal() or clone.levels_completed > env.levels_completed:
            break
        prior = condensed
        clone.step(5)

    staged6 = replay(env, picked + stage6)
    print(
        "PORT6_DISMISS_VARIANTS",
        tuple(
            (path, state(child))
            for path, child in dismissal_variants(staged6)
        ),
        flush=True,
    )
    pre_stage6 = replay(env, picked + stage6[:-1])
    print(
        "PORT6_COMBINED",
        combined_stage_dismiss(pre_stage6),
        flush=True,
    )

    place_staged = [4, 1, 5, 1, 1, 5, 2]
    staged_candidate = picked + stage6 + dismiss6 + place_staged
    clone = replay(env, staged_candidate)
    print(
        "PORT6_STAGED", len(staged_candidate), state(clone), flush=True
    )
    prior = None
    for turn in range(13 + len(staged_candidate), 71):
        current = state(clone)
        condensed = (
            current["empty"],
            current["filled"],
            current["helper"],
            current["level"],
            current["terminal"],
        )
        if condensed != prior:
            print("PORT6_STAGED_EVENT", turn, current, flush=True)
        if clone.terminal() or clone.levels_completed > env.levels_completed:
            break
        prior = condensed
        clone.step(5)

    fill_around_stage = (
        [3] * 4 + [5]
        + [2] + [4] * 6 + [1] * 3 + [3, 1, 5]
    )
    around_candidate = picked + stage6 + dismiss6 + fill_around_stage
    clone = replay(env, around_candidate)
    print(
        "PORT6_AROUND", len(around_candidate), state(clone), flush=True
    )
    print("PORT6_AROUND_FINISH", short_finish(clone), flush=True)

    fill_bottom_vertical = (
        [1] * 2 + [3] * 4 + [2, 5]
        + [1] * 3 + [4] * 5 + [2, 5]
    )
    vertical_candidate = (
        picked + stage6 + dismiss6 + fill_bottom_vertical
    )
    clone = replay(env, vertical_candidate)
    print(
        "PORT6_VERTICAL", len(vertical_candidate), state(clone), flush=True
    )
    print("PORT6_VERTICAL_FINISH", short_finish(clone), flush=True)

    fill_middle_from_below = (
        [2] + [3] * 4 + [1, 5]
        + [4] * 6 + [1] * 4 + [5]
    )
    middle_candidate = (
        picked + stage6 + dismiss6 + fill_middle_from_below
    )
    clone = replay(env, middle_candidate)
    print(
        "PORT6_MIDDLE", len(middle_candidate), state(clone), flush=True
    )
    print("PORT6_MIDDLE_FINISH", short_finish(clone), flush=True)

    fill_bottom_reoriented = (
        [3] * 4 + [5]
        + [4] * 2 + [5, 2, 3, 1, 5]
        + [1] * 3 + [4] * 3 + [5]
    )
    reoriented_candidate = (
        picked + stage6 + dismiss6 + fill_bottom_reoriented
    )
    clone = replay(env, reoriented_candidate)
    print(
        "PORT6_REORIENTED",
        len(reoriented_candidate),
        state(clone),
        flush=True,
    )
    print("PORT6_REORIENTED_FINISH", short_finish(clone), flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
