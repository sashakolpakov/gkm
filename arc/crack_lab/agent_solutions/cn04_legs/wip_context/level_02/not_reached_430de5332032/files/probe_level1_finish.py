"""Reconstruct the rewarded level-1 rotation from observed silhouettes."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception


def cells(frame, color):
    a = perception.arr(frame)
    return {
        (r // 3, c // 3)
        for r in range(3, 61, 3)
        for c in range(3, 61, 3)
        if int(a[r, c]) == color
    }


def normalize(points):
    r0 = min(r for r, _ in points)
    c0 = min(c for _, c in points)
    return sorted((r - r0, c - c0) for r, c in points)


def probe(env):
    initial = env.clone()
    prefinish = perception.replay(env, [2] * 7 + [4] * 4)
    print("initial_A", normalize(cells(initial.frame(), 0)))
    for turns in (1, 2, 3):
        rotated = perception.replay(env, [5] * turns)
        print("rotated_A", turns, normalize(cells(rotated.frame(), 0)))
    print("prefinish_A", sorted(cells(prefinish.frame(), 0)))
    print("target_B", sorted(cells(prefinish.frame(), 14)))
    print("pegs", sorted(cells(prefinish.frame(), 8)))
    original_pegs = sorted(cells(env.frame(), 8))
    visited = set()
    known = [2] * 7 + [4] * 4 + [5] * 3
    for length in range(len(known) + 1):
        node = perception.replay(env, known[:length])
        covered = {p for p in original_pegs if p in cells(node.frame(), 0)}
        new = covered - visited
        visited |= covered
        if new or length == len(known):
            print(
                "coverage",
                length,
                "new",
                sorted(new),
                "visited",
                len(visited),
                "level",
                node.levels_completed,
            )
    for turns in (1, 2, 3):
        final = perception.replay(prefinish, [5] * turns)
        print(
            "after_rotate",
            turns,
            "level",
            final.levels_completed,
            "A",
            [] if final.levels_completed else sorted(cells(final.frame(), 0)),
        )


arena.run_program("cn04", probe)
