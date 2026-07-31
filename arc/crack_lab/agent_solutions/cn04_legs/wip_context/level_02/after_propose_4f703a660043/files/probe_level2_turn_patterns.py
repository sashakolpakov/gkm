"""Enumerate compact peg-coverage patterns from repeated rotations."""
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
from players import play_level_1
from probe_level2_collect import avatar_key, covered, peg_centers


GROUPS = ((1, 3), (0, 2, 4, 5), (6, 7, 8, 11), (9, 10))


def probe(env):
    play_level_1(env)
    root = env.clone()
    pegs = peg_centers(root.frame())
    q = deque([(root.clone(), [])])
    seen = {avatar_key(root)}
    best = {i: (-1, None, None) for i in range(len(GROUPS))}
    while q and len(seen) < 3000:
        node, path = q.popleft()
        for group_no, group in enumerate(GROUPS):
            child = node.clone()
            pattern = []
            union = set()
            for _ in range(4):
                child.step(5)
                hit = tuple(i for i in group if covered(child, pegs[i]))
                pattern.append(hit)
                union.update(hit)
            quality = (len(union), max(map(len, pattern)))
            old = best[group_no][0]
            old_pair = best[group_no][1]
            if quality > (old, -1 if old_pair is None else old_pair):
                best[group_no] = (quality[0], quality[1], (path, tuple(pattern)))
        if len(path) >= 45:
            continue
        for action in (1, 2, 3, 4, 5):
            child = node.clone()
            child.step(action)
            key = avatar_key(child)
            if key not in seen:
                seen.add(key)
                q.append((child, path + [action]))
    print("states", len(seen), "pegs", list(enumerate(pegs)))
    for group_no in range(len(GROUPS)):
        print("group", group_no, "best", best[group_no])


arena.run_program("cn04", probe)
