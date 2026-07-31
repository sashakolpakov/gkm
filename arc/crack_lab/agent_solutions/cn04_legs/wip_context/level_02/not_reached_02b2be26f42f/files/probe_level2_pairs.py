"""List simultaneously coverable peg pairs and shortest observed paths."""
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

from players import play_level_1
from probe_level2_collect import avatar_key, covered, peg_centers
GROUPS = ((1, 3), (0, 2, 4, 5), (6, 7, 8, 11), (9, 10))


def probe(env):
    play_level_1(env)
    root = env.clone()
    pegs = peg_centers(root.frame())
    q = deque([(root.clone(), [])])
    seen = {avatar_key(root)}
    pairs = {n: {} for n in range(len(GROUPS))}
    while q and len(seen) < 2000:
        node, path = q.popleft()
        for n, group in enumerate(GROUPS):
            hit = tuple(i for i in group if covered(node, pegs[i]))
            if len(hit) >= 2 and hit not in pairs[n]:
                pairs[n][hit] = path
        if len(path) >= 45:
            continue
        for action in (1, 2, 3, 4, 5):
            child = node.clone()
            child.step(action)
            key = avatar_key(child)
            if key not in seen:
                seen.add(key)
                q.append((child, path + [action]))
    print("states", len(seen))
    for n in pairs:
        print("group", n, sorted(pairs[n].items()))


arena.run_program("cn04", probe)
