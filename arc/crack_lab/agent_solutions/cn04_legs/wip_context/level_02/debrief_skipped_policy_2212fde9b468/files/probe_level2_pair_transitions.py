"""Observe which peg-pair socket states are connected by one rotation."""
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from players import play_level_1
from probe_level2_collect import avatar_key, covered, peg_centers


GROUPS = ((1, 3), (0, 2, 4, 5), (6, 7, 8, 11), (9, 10))


def hit_pair(node, group, pegs):
    hit = tuple(i for i in group if covered(node, pegs[i]))
    return hit if len(hit) >= 2 else ()


def probe(env):
    play_level_1(env)
    root = env.clone()
    pegs = peg_centers(root.frame())
    q = deque([root.clone()])
    seen = {avatar_key(root)}
    transitions = {n: set() for n in range(len(GROUPS))}
    while q and len(seen) < 2000:
        node = q.popleft()
        rotated = node.clone()
        rotated.step(5)
        for n, group in enumerate(GROUPS):
            before = hit_pair(node, group, pegs)
            after = hit_pair(rotated, group, pegs)
            if before or after:
                transitions[n].add((before, after))
        for action in (1, 2, 3, 4, 5):
            child = node.clone()
            child.step(action)
            key = avatar_key(child)
            if key not in seen:
                seen.add(key)
                q.append(child)
    print("states", len(seen))
    for n in transitions:
        print("group", n, sorted(transitions[n]))


arena.run_program("cn04", probe)
