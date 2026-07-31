"""Locate the context that enables the four-state selector."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve


TOP = (6, 54, 6)
MAIN = (6, 50, 26)
NEW = (6, 50, 46)
TO_BRIDGE = [3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 2, 3]


def pad(env):
    return tuple(
        int(value)
        for value in perception.arr(env.frame())[48:50, 18:20].flat
    )


def enabled(node):
    clone = node.clone()
    before = pad(clone)
    clone.step(*NEW)
    return before, pad(clone)


def observe(env):
    solve.solve(env)
    print("INITIAL", enabled(env))
    main_only = env.clone()
    main_only.step(*MAIN)
    print("MAIN_ONLY", enabled(main_only))
    transfer = env.clone()
    for _ in range(4):
        transfer.step(*TOP)
    for action in TO_BRIDGE:
        transfer.step(action)
    transfer.step(*TOP)
    transfer.step(1)
    transfer.step(3)
    transfer.step(*TOP)
    print("TRANSFER", enabled(transfer))
    for _ in range(6):
        transfer.step(1)
    print("LOWER_ISLAND", enabled(transfer))
    transfer.step(*MAIN)
    print("MAIN_VERTICAL", enabled(transfer))
    for step_count in range(1, 15):
        transfer.step(1)
        before, after = enabled(transfer)
        if before != after:
            print("UNLOCK_AT", step_count, before, after)
            break


arena.run_program("dc22", observe)
