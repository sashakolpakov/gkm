"""Test direct hand-delivery and early dismissal on level 7."""

from collections import deque

import gkm_try

from perception import arr
from probe_minimize_segments import CaptureSegments
from probe9_verify import boxes, tile_map


def target_filled(frame):
    grid = arr(frame)
    return tuple(
        (row, 3)
        for row in (7, 8)
        if 4 in set(
            int(value)
            for value in grid[row * 4 : row * 4 + 4, 12:16].flat
        )
    )


def target_signatures(frame):
    grid = arr(frame)
    return tuple(
        tuple(
            sorted(
                set(
                    int(value)
                    for value in grid[
                        row * 4 : row * 4 + 4, 12:16
                    ].flat
                )
            )
        )
        for row in (7, 8)
    )


def state(env):
    return {
        "level": env.levels_completed,
        "avatar": boxes(env.frame(), 14),
        "cargo": boxes(env.frame(), 4),
        "courier": boxes(env.frame(), 15),
        "filled": target_filled(env.frame()),
        "target": target_signatures(env.frame()),
    }


def replay(base, route):
    child = base.clone()
    for action in route:
        if child.terminal() or child.levels_completed > base.levels_completed:
            break
        child.step(action)
    return child


def shortest_dismissal(base, max_depth=8):
    queue = deque([(base.clone(), ())])
    seen = {arr(base.frame()).tobytes()}
    while queue:
        node, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for action in node.actions:
            child = node.clone()
            child.step(action)
            child_path = path + (int(action),)
            if not boxes(child.frame(), 15):
                return child_path, state(child)
            key = arr(child.frame()).tobytes()
            if key not in seen:
                seen.add(key)
                queue.append((child, child_path))
    return None, len(seen)


def inspect(env):
    capture = CaptureSegments(env)
    gkm_try.resumed_solve(capture)
    start = capture.starts[6]
    candidates = {
        "lower_direct": [4, 4, 5] + [3] * 4 + [5],
        "upper_direct": (
            [1] * 2 + [4] * 3 + [5, 2] + [3] * 5 + [5]
        ),
        "upper_from_right": (
            [4] * 5 + [1] * 2 + [4, 5]
            + [2] + [3] * 5 + [5]
        ),
        "meet_use": [4, 4, 5],
        "meet_wait_use": [4, 4, 5, 5],
    }
    print("L7_DIRECT_START", state(start), flush=True)
    for label, route in candidates.items():
        child = replay(start, route)
        print(
            "L7_DIRECT",
            label,
            len(route),
            state(child),
            flush=True,
        )
        if label in ("lower_direct", "upper_direct", "upper_from_right"):
            print(label, *tile_map(child.frame()), sep="\n", flush=True)
    upper_pick = [4] * 5 + [1] * 2 + [4, 5] + [2]
    traced = start.clone()
    for turn, action in enumerate(upper_pick, 1):
        traced.step(action)
        print("L7_PICK_TRACE", turn, action, state(traced), flush=True)
    for left_steps in range(2, 7):
        child = replay(start, upper_pick + [3] * left_steps + [5])
        print(
            "L7_UPPER_DROP",
            left_steps,
            len(upper_pick) + left_steps + 1,
            state(child),
            flush=True,
        )
    upper_route = candidates["upper_from_right"]
    dismissals = (
        [4, 4, 5],
        [4, 4, 4, 5],
        [4, 5],
        [4, 4, 5, 5],
        [1] + [4] * 2 + [2, 5],
        [1] + [4] * 3 + [2, 5],
        [1] + [4] * 4 + [2, 5],
    )
    for dismiss in dismissals:
        child = replay(start, upper_route + dismiss)
        print(
            "L7_AFTER_UPPER",
            dismiss,
            len(upper_route + dismiss),
            state(child),
            flush=True,
        )
    fast_upper = candidates["upper_direct"]
    for right_steps in range(2, 8):
        dismiss = [1] + [4] * right_steps + [2, 5]
        child = replay(start, fast_upper + dismiss)
        print(
            "L7_FAST_DISMISS",
            right_steps,
            len(fast_upper + dismiss),
            state(child),
            flush=True,
        )
    for uses in range(1, 5):
        dismiss = [1] + [4] * 2 + [2] + [5] * uses
        child = replay(start, fast_upper + dismiss)
        print(
            "L7_FAST_USE",
            uses,
            len(fast_upper + dismiss),
            state(child),
            flush=True,
        )
    fast_contacts = (
        [1, 4, 4, 2, 4, 5],
        [1, 4, 4, 2, 5, 4, 5],
        [1, 4, 4, 2, 4, 4, 5],
        [1, 4, 4, 2, 1, 4, 5],
        [1, 4, 4, 2, 4, 1, 5],
        [1, 4, 4, 2, 4, 5, 1, 5],
        [1, 4, 4, 2, 4, 1, 5, 5],
    )
    for contact in fast_contacts:
        child = replay(start, fast_upper + contact)
        print(
            "L7_FAST_CONTACT",
            contact,
            len(fast_upper + contact),
            state(child),
            flush=True,
        )
    good_prefix = upper_route + [1] + [4] * 2 + [2, 5]
    best = None
    for right_steps in range(6, 9):
        for left_steps in range(8, 11):
            for clear in ((), (1,), (2,), (3,), (4,)):
                suffix = (
                    [4] * right_steps
                    + [2] * 2
                    + [4, 5]
                    + [3] * left_steps
                    + [5]
                    + list(clear)
                )
                child = replay(start, good_prefix + suffix)
                score = (
                    child.levels_completed - start.levels_completed,
                    sum(
                        1
                        for signature in target_signatures(child.frame())
                        if 4 in signature
                    ),
                )
                candidate = (
                    score,
                    len(good_prefix + suffix),
                    right_steps,
                    left_steps,
                    clear,
                    state(child),
                )
                if best is None or (score, -candidate[1]) > (
                    best[0],
                    -best[1],
                ):
                    best = candidate
    print("L7_FULL_BEST", best, flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
