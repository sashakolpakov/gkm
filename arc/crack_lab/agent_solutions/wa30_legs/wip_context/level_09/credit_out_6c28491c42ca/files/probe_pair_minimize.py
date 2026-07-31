"""Shorten proven level routes with coordinated run-count reductions."""

import os

import gkm_try

from probe_minimize_segments import CaptureSegments, evaluate


def encode(route):
    runs = []
    for action in route:
        if runs and runs[-1][0] == action:
            runs[-1][1] += 1
        else:
            runs.append([action, 1])
    return runs


def expand(runs):
    return [
        action
        for action, count in runs
        for _ in range(count)
    ]


def pair_minimize(start, route):
    best = list(route)
    won, best_turns = evaluate(start, best, len(best))
    if not won:
        return best, None
    passes = 0
    while passes < 12:
        passes += 1
        runs = encode(best)
        accepted = False
        for first in range(len(runs)):
            for second in range(first, len(runs)):
                if first == second and runs[first][1] < 2:
                    continue
                candidate_runs = [run[:] for run in runs]
                candidate_runs[first][1] -= 1
                candidate_runs[second][1] -= 1
                candidate_runs = [run for run in candidate_runs if run[1]]
                candidate = expand(candidate_runs)
                won, turns = evaluate(start, candidate, best_turns - 1)
                if won:
                    best = candidate
                    best_turns = turns
                    print(
                        "PAIR_ACCEPT",
                        passes,
                        first,
                        second,
                        len(best),
                        best_turns,
                        encode(best),
                        flush=True,
                    )
                    accepted = True
                    break
            if accepted:
                break
        if not accepted:
            break
    return best, best_turns


def triple_minimize(start, route):
    best = list(route)
    won, best_turns = evaluate(start, best, len(best))
    if not won:
        return best, None
    for pass_number in range(1, 7):
        runs = encode(best)
        eligible = [
            index for index, (_, count) in enumerate(runs) if count > 1
        ]
        accepted = False
        for first_offset, first in enumerate(eligible):
            for second_offset in range(first_offset, len(eligible)):
                second = eligible[second_offset]
                for third_offset in range(second_offset, len(eligible)):
                    third = eligible[third_offset]
                    decrements = {
                        index: (first, second, third).count(index)
                        for index in (first, second, third)
                    }
                    if any(
                        runs[index][1] <= amount
                        for index, amount in decrements.items()
                    ):
                        continue
                    candidate_runs = [run[:] for run in runs]
                    for index, amount in decrements.items():
                        candidate_runs[index][1] -= amount
                    candidate = expand(candidate_runs)
                    won, turns = evaluate(start, candidate, best_turns - 1)
                    if won:
                        best = candidate
                        best_turns = turns
                        print(
                            "TRIPLE_ACCEPT",
                            pass_number,
                            first,
                            second,
                            third,
                            len(best),
                            best_turns,
                            encode(best),
                            flush=True,
                        )
                        accepted = True
                        break
                if accepted:
                    break
            if accepted:
                break
        if not accepted:
            break
    return best, best_turns


def inspect(env):
    capture = CaptureSegments(env)
    gkm_try.resumed_solve(capture)
    segments = {level + 1: route for level, route in capture.transitions}
    level = int(os.environ.get("GKM_OPT_LEVEL", "5"))
    route = segments[level]
    print("PAIR_BASE", level, len(route), encode(route), flush=True)
    mode = os.environ.get("GKM_OPT_MODE", "pair")
    minimize = triple_minimize if mode == "triple" else pair_minimize
    best, turns = minimize(capture.starts[level - 1], route)
    print(
        "PAIR_RESULT",
        level,
        len(best),
        turns,
        encode(best),
        best,
        flush=True,
    )


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
