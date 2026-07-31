"""Compact traces of target and peg coverage along known/candidate paths."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
from players import play_level_1


def cells(frame, color):
    a = perception.arr(frame)
    return {
        (r // 3, c // 3)
        for r in range(3, 61, 3)
        for c in range(3, 61, 3)
        if int(a[r, c]) == color
    }


def trace(env, path, colors):
    base_level = env.levels_completed
    targets = {color: cells(env.frame(), color) for color in colors}
    pegs = cells(env.frame(), 8)
    seen = {color: set() for color in colors}
    seen_pegs = set()
    for index, action in enumerate(path, 1):
        env.step(action)
        if env.levels_completed > base_level:
            print("step", index, action, "level", env.levels_completed)
            break
        ours = cells(env.frame(), 0)
        hits = {}
        for color, target in targets.items():
            new = (ours & target) - seen[color]
            seen[color] |= ours & target
            if new:
                hits[color] = (len(new), len(seen[color]), len(target))
        new_pegs = (ours & pegs) - seen_pegs
        seen_pegs |= ours & pegs
        if hits or new_pegs or action == 5:
            print(
                "step", index, action, "hits", hits,
                "pegs", (len(new_pegs), len(seen_pegs), len(pegs)),
            )


def probe(env):
    print("level1")
    trace(env.clone(), [2] * 7 + [4] * 4 + [5] * 3, (14,))
    play_level_1(env)
    print("level2_direct_D")
    trace(
        env.clone(),
        [2] * 10 + [4] * 10 + [5] * 2 + [4] * 2,
        (14, 11, 9),
    )


arena.run_program("cn04", probe)
