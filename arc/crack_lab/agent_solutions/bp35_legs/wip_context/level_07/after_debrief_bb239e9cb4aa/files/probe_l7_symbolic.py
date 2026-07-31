"""Symbolic BFS for the reproduced hazard/support chamber."""

from collections import deque


BOARD = [
    "########",
    "########",
    "#####x#P",
    "#####x#.",
    "#####x..",
    "#####x..",
    "..P.Px#.",
    ".xx.Xx#.",
    ".x.X.x#.",
    ".x.P.P#.",
]

START = (6, 0, -1)  # row, column, gravity (-1 up, +1 down)
SUPPORTS = [
    (r, c)
    for r, row in enumerate(BOARD)
    for c, value in enumerate(row)
    if value in "xX"
]
INDEX = {cell: index for index, cell in enumerate(SUPPORTS)}
START_MASK = sum(
    1 << INDEX[(r, c)]
    for r, row in enumerate(BOARD)
    for c, value in enumerate(row)
    if value == "X"
)


def tile(r, c, mask):
    value = BOARD[r][c]
    if value in "xX":
        return "X" if mask & (1 << INDEX[(r, c)]) else "."
    return value


def settle(r, c, gravity, mask):
    while True:
        nr = r + gravity
        if nr < 0 or nr >= len(BOARD):
            return "exit", r, c, gravity
        value = tile(nr, c, mask)
        if value in "#X":
            return r, c, gravity, mask
        if value == "P":
            return None
        r = nr


def successors(state):
    r, c, gravity, mask = state
    for dc, name in ((-1, "L"), (1, "R")):
        nc = c + dc
        if not 0 <= nc < 8 or tile(r, nc, mask) in "#XP":
            continue
        child = settle(r, nc, gravity, mask)
        if child is not None:
            yield name, child
    child = settle(r, c, -gravity, mask)
    if child is not None:
        yield "G", child
    for cell, index in INDEX.items():
        if cell == (r, c):
            continue
        yield f"T{cell[0]}{cell[1]}", (r, c, gravity, mask ^ (1 << index))


def goal(state):
    r, c, _gravity, _mask = state
    return c == 5 and r <= 2


start = (*START, START_MASK)
queue = deque([(start, ())])
seen = {start}
positions = {start[:3]}
exits = []
while queue:
    state, path = queue.popleft()
    if goal(state):
        print({"steps": len(path), "state": state[:3], "path": path})
        break
    if len(path) >= 28:
        continue
    for action, child in successors(state):
        if child[0] == "exit":
            exits.append((child[3], child[2], path + (action,)))
            continue
        if child in seen:
            continue
        seen.add(child)
        positions.add(child[:3])
        queue.append((child, path + (action,)))
else:
    best_exits = {}
    for direction, column, path in exits:
        best_exits.setdefault((direction, column), path)
        if len(path) < len(best_exits[(direction, column)]):
            best_exits[(direction, column)] = path
    print(
        {
            "found": False,
            "seen": len(seen),
            "positions": sorted(positions),
            "exits": sorted(
                (direction, column, path)
                for (direction, column), path in best_exits.items()
            ),
        }
    )
