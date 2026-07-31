"""Enumerate next-region peg pairs reachable by each relay agent."""
import sys
from collections import deque
from itertools import combinations

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from players import play_level_1
from probe_level2_collect import avatar_key, covered, peg_centers


SELECT = {"A": (18, 21), "B": (45, 15), "C": (18, 39)}
TARGET = {"A": (0, 2, 4, 5), "B": (6, 7, 8, 11), "C": (9, 10)}


def probe(env):
    play_level_1(env)
    pegs = peg_centers(env.frame())
    for name in "ABC":
        root = env.clone()
        root.step(6, *SELECT[name])
        q = deque([(root, [])])
        seen = {avatar_key(root)}
        found = {}
        while q and len(seen) < 3000:
            node, path = q.popleft()
            hit = tuple(i for i in TARGET[name] if covered(node, pegs[i]))
            for pair in combinations(hit, 2):
                found.setdefault(pair, path)
            for action in (1, 2, 3, 4, 5):
                child = node.clone()
                child.step(action)
                key = avatar_key(child)
                if key not in seen:
                    seen.add(key)
                    q.append((child, path + [action]))
        print("agent", name, "states", len(seen), "pairs", sorted(found.items()))


arena.run_program("cn04", probe)
