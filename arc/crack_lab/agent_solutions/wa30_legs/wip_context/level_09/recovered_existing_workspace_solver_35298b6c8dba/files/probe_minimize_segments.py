"""Greedily shorten proven per-level action segments on pristine clones."""

import gkm_try


LEVELS = (5, 3, 8, 4, 7, 2, 6, 1)


class CaptureSegments:
    def __init__(self, env):
        self.env = env
        self.actions_seen = []
        self.starts = {env.levels_completed: env.clone()}
        self.transitions = []
        self.last_transition_index = 0

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action, *args):
        before = self.env.levels_completed
        result = self.env.step(action, *args)
        self.actions_seen.append(action)
        if self.env.levels_completed > before:
            end = len(self.actions_seen)
            self.transitions.append(
                (
                    before,
                    tuple(self.actions_seen[self.last_transition_index:end]),
                )
            )
            self.last_transition_index = end
            self.starts[self.env.levels_completed] = self.env.clone()
        return result


def evaluate(start, actions, max_turns):
    clone = start.clone()
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
    return clone.levels_completed > base_level, used


def minimize(start, route):
    best = list(route)
    won, best_turns = evaluate(start, best, len(best))
    if not won:
        return best, None
    for chunk in (16, 8, 4, 2, 1):
        changed = True
        while changed:
            changed = False
            index = 0
            while index < len(best):
                candidate = best[:index] + best[index + chunk:]
                won, turns = evaluate(
                    start,
                    candidate,
                    max(0, best_turns - 1),
                )
                if won and turns < best_turns:
                    print(
                        "SEGMENT_ACCEPT",
                        start.levels_completed + 1,
                        chunk,
                        index,
                        len(candidate),
                        turns,
                        flush=True,
                    )
                    best = candidate
                    best_turns = turns
                    changed = True
                else:
                    index += chunk
    return best, best_turns


def inspect(env):
    capture = CaptureSegments(env)
    gkm_try.resumed_solve(capture)
    segments = {level + 1: route for level, route in capture.transitions}
    print(
        "SEGMENT_BASE",
        tuple((level, len(route)) for level, route in sorted(segments.items())),
        flush=True,
    )
    for level in LEVELS:
        route = segments[level]
        best, turns = minimize(capture.starts[level - 1], route)
        print(
            "SEGMENT_RESULT",
            level,
            {
                "original": len(route),
                "route_length": len(best),
                "turns": turns,
                "route": best,
            },
            flush=True,
        )


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
