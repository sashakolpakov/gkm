import itertools
from collections import deque

import gkm_try as harness


GRAPH = {
    "N": ((3, "NW"), (4, "NE")),
    "NW": ((2, "W"), (4, "N")),
    "NE": ((2, "E"), (3, "N")),
    "W": ((1, "NW"), (2, "SW")),
    "E": ((1, "NE"), (2, "SE")),
    "SW": ((1, "W"), (4, "S")),
    "SE": ((1, "E"), (3, "S")),
    "S": ((3, "SW"), (4, "SE")),
}
CLICKS = {15: (28, 4), 12: (34, 4), 14: (46, 4), 8: (52, 4)}
CANDIDATES = {
    15: ("NW",),
    12: ("N", "NW", "NE"),
    14: ("E", "SE"),
    8: ("W", "SW", "SE", "S"),
}


def routes():
    out = {}
    for start in GRAPH:
        q = deque([(start, ())])
        seen = {start}
        while q:
            position, path = q.popleft()
            out[start, position] = path
            for action, nxt in GRAPH[position]:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, path + (action,)))
    return out


def observe(env):
    harness.m.solve(env)
    route = routes()
    tried = 0
    for order in itertools.permutations(CANDIDATES):
        choices = itertools.product(*(CANDIDATES[color] for color in order))
        for masks in choices:
            node = env.clone()
            position = "N"
            recipe = []
            for color, mask in zip(order, masks):
                moves = route[position, mask]
                for action in moves:
                    node.step(action)
                node.step(6, *CLICKS[color])
                node.step(5)
                recipe.append((color, mask, moves))
                position = mask
                if node.levels_completed >= 3:
                    print("FOUND", recipe, "TRIED", tried + 1)
                    return
            tried += 1
    print("NOT_FOUND", tried)


if __name__ == "__main__":
    harness.A.run_program("cd82", observe)
