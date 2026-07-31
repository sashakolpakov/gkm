"""Search for a one-turn-faster combined dismissal and local pickup."""

import gkm_try

from perception import arr
from probe9_candidate import direct_second_prefix
from probe9_verify import (
    ReachedLevel9,
    StopAtLevel9,
    boxes,
    target_state,
)


def avatar_cell(frame):
    avatar = boxes(frame, 14)
    if not avatar:
        return None
    row0, col0, row1, col1 = avatar[0]
    return ((row0 + row1) // 8, (col0 + col1) // 8)


def local_cargo_absent(frame):
    grid = arr(frame)
    return 4 not in grid[32:36, 4:8]


def goal(env):
    target = target_state(env.frame())
    return (
        (5, 6) in target["filled"]
        and not boxes(env.frame(), 15)
        and avatar_cell(env.frame()) == (8, 2)
        and local_cargo_absent(env.frame())
    )


def search(env, max_depth=8, max_transitions=20000):
    frontier = [(env.clone(), [])]
    transitions = 0
    for depth in range(1, max_depth + 1):
        next_states = {}
        for node, path in frontier:
            for action in node.actions:
                child = node.clone()
                child.step(action)
                child_path = path + [action]
                transitions += 1
                if goal(child):
                    print(
                        "PICKSEARCH_WIN",
                        depth,
                        transitions,
                        child_path,
                        flush=True,
                    )
                    return child_path
                if not child.terminal():
                    next_states.setdefault(
                        arr(child.frame()).tobytes(),
                        (child, child_path),
                    )
                if transitions >= max_transitions:
                    print("PICKSEARCH_LIMIT", transitions, flush=True)
                    return None
        frontier = list(next_states.values())
        print(
            "PICKSEARCH_DEPTH",
            depth,
            len(frontier),
            transitions,
            flush=True,
        )
    return None


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel9(env))
    except ReachedLevel9:
        pass

    state = env.clone()
    prefix = direct_second_prefix()
    for action in prefix:
        state.step(action)
    print(
        "PICKSEARCH_START",
        len(prefix),
        avatar_cell(state.frame()),
        boxes(state.frame(), 15),
        flush=True,
    )
    path = search(state)
    print("PICKSEARCH_RESULT", path, flush=True)
    if path is None:
        return
    reached = state.clone()
    for action in path:
        reached.step(action)
    grid = arr(reached.frame())
    print(
        "PICKSEARCH_REACHED",
        {
            "avatar": boxes(reached.frame(), 14),
            "cargo": boxes(reached.frame(), 4),
            "cells": {
                (row, col): tuple(sorted(set(int(value) for value in
                    grid[row * 4:row * 4 + 4,
                         col * 4:col * 4 + 4].flat)))
                for row in (7, 8)
                for col in range(4)
            },
        },
        flush=True,
    )
    for action in reached.actions:
        child = reached.clone()
        child.step(action)
        print(
            "PICKSEARCH_ACTION",
            action,
            boxes(child.frame(), 14),
            boxes(child.frame(), 4),
            flush=True,
        )


gkm_try.A.run_program("wa30", inspect)
