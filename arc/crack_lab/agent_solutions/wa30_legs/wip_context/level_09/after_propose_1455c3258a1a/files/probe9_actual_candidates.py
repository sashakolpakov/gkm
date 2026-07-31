"""Reproduce compact candidate routes from the real level-9 handoff."""

import gkm_try

from probe9_verify import boxes, target_state


def state(env):
    target = target_state(env.frame())
    return {
        "level": env.levels_completed,
        "terminal": env.terminal(),
        "avatar": boxes(env.frame(), 14),
        "cargo": boxes(env.frame(), 4),
        "helper": boxes(env.frame(), 12),
        "thief": boxes(env.frame(), 15),
        "empty": target["empty"],
        "filled": target["filled"],
    }


def replay(base, actions):
    clone = base.clone()
    for action in actions:
        if clone.terminal() or clone.levels_completed > base.levels_completed:
            break
        clone.step(action)
    return clone


def shortest_dismissal(start, label, max_depth=5):
    frontier = [(start.clone(), [])]
    seen = {start.frame().tobytes()}
    transitions = 0
    for depth in range(1, max_depth + 1):
        next_frontier = []
        wins = []
        for node, path in frontier:
            for action in node.actions:
                child = node.clone()
                child.step(action)
                transitions += 1
                child_path = path + [action]
                if not boxes(child.frame(), 15):
                    wins.append((child_path, state(child)))
                    continue
                key = child.frame().tobytes()
                if key not in seen:
                    seen.add(key)
                    next_frontier.append((child, child_path))
        print(
            "DISMISS_DEPTH", label, depth, len(next_frontier), transitions,
            flush=True,
        )
        if wins:
            return wins
        frontier = next_frontier
    return []


def inspect(env):
    gkm_try.resumed_solve(env)

    first_delivery = (
        [3, 1, 5]
        + [3] * 6 + [1] * 3 + [3, 5]
    )
    second_delivery = (
        [2] * 2 + [4] * 5 + [1, 5]
        + [2] + [3] * 6 + [1] * 3 + [5]
    )
    second_pick = [2] * 2 + [4] * 5 + [1, 5]
    prefix = first_delivery + second_delivery
    inline_dismiss = prefix[:30] + [5, 3, 1, 5]
    replace_second = [4, 1, 5] + [1] * 3 + [5, 2]
    early_inline_a = prefix[:29] + [2, 5, 4, 3, 5]
    early_inline_b = prefix[:29] + [5, 3, 2, 1, 5]
    stage8 = first_delivery + second_pick + [2] + [3] * 4 + [1, 5]
    stage8_dismiss = stage8 + [3, 3, 3, 1, 5]
    stage8_manual = (
        stage8_dismiss
        + [3] * 4 + [5]
        + [1] * 3 + [4] * 6 + [5, 4]
    )
    paths = {
        "first_delivery": first_delivery,
        "two_deliveries": prefix,
        "two_idle23": prefix + [5] * 23,
        "old_dismiss": prefix + [2, 2, 3, 5],
        "old_tail": (
            prefix + [2, 2, 3, 5]
            + [3] * 3 + [5, 4]
            + [1] * 6 + [4] * 5
            + [5, 1, 3, 2, 5, 2, 5, 1]
        ),
        "inline_dismiss": inline_dismiss,
        "inline_idle": inline_dismiss + [5] * 24,
        "inline_replaced": inline_dismiss + replace_second,
        "inline_finish": inline_dismiss + replace_second + [5] * 16,
        "early_a_idle": early_inline_a + [5] * 24,
        "early_b_idle": early_inline_b + [5] * 24,
        "stage8_idle": stage8 + [5] * 30,
        "stage8_safe_idle": stage8_dismiss + [5] * 24,
        "stage8_manual": stage8_manual + [5] * 8,
    }
    for name, path in paths.items():
        clone = replay(env, path)
        print("CANDIDATE", name, len(path), state(clone), flush=True)

    watch = replay(env, inline_dismiss + replace_second)
    for turn in range(55, 71):
        current = state(watch)
        print(
            "HELPER_TRACE",
            turn,
            current["helper"],
            current["empty"],
            current["filled"],
            flush=True,
        )
        if watch.terminal():
            break
        watch.step(5)

    trace = env.clone()
    prior = None
    for turn, action in enumerate(prefix + [5] * 23, 14):
        trace.step(action)
        current = state(trace)
        condensed = (
            current["empty"],
            current["filled"],
            current["thief"],
            current["level"],
            current["terminal"],
        )
        if action == 5 or condensed != prior:
            print("EVENT", turn, action, current, flush=True)
        prior = condensed
        if trace.terminal() or trace.levels_completed > env.levels_completed:
            break


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
