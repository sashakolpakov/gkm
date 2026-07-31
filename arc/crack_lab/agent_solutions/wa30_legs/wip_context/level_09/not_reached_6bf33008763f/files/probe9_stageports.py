"""Probe staging the second remote cargo on the surviving courier's row."""

import gkm_try

from perception import arr
from probe9_reroute import second_pick_prefix
from probe9_verify import (
    ReachedLevel9,
    StopAtLevel9,
    boxes,
    summary,
    target_state,
)


def shortest_dismiss(env, max_depth=6):
    frontier = [(env.clone(), [])]
    for depth in range(1, max_depth + 1):
        next_states = {}
        for node, path in frontier:
            for action in node.actions:
                child = node.clone()
                child.step(action)
                child_path = path + [action]
                if not boxes(child.frame(), 15):
                    return child, child_path
                next_states.setdefault(
                    arr(child.frame()).tobytes(),
                    (child, child_path),
                )
        frontier = list(next_states.values())
    return None, None


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel9(env))
    except ReachedLevel9:
        pass

    picked = env.clone()
    prefix = second_pick_prefix()
    for action in prefix:
        picked.step(action)

    for column in range(4, 9):
        stage_path = [2] + [3] * (12 - column) + [1, 5]
        staged = picked.clone()
        for action in stage_path:
            staged.step(action)
        stage_turn = len(prefix) + len(stage_path)
        print(
            "PORT_STAGE",
            column,
            summary(staged, stage_turn),
            flush=True,
        )
        dismissed, contact = shortest_dismiss(staged)
        print(
            "PORT_DISMISS",
            column,
            contact,
            None if dismissed is None else summary(
                dismissed, stage_turn + len(contact)
            ),
            flush=True,
        )
        if dismissed is None:
            continue
        idle = dismissed.clone()
        idle_turn = stage_turn + len(contact)
        prior = None
        while not idle.terminal():
            target = target_state(idle.frame())
            condensed = (target["empty"], target["filled"])
            if condensed != prior:
                print(
                    "PORT_IDLE",
                    column,
                    idle_turn,
                    condensed,
                    boxes(idle.frame(), 12),
                    flush=True,
                )
            prior = condensed
            idle.step(5)
            idle_turn += 1


gkm_try.A.run_program("wa30", inspect)
