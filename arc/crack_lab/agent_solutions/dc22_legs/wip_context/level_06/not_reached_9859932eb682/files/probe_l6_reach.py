"""Static movement reachability for each observed control phase."""
from collections import deque
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve


TOP = (6, 54, 6)
MAIN = (6, 50, 26)


def avatar_position(env):
    for blob in perception.connected_components(
            env.frame(), colors=(14,), min_area=2):
        if blob.bbox[1] < 32:
            return blob.top_left
    return None


def movement_reach(root):
    start = avatar_position(root)
    queue = deque([(root.clone(), [])])
    seen = {start: []}
    while queue:
        node, path = queue.popleft()
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            position = avatar_position(child)
            if child.levels_completed > root.levels_completed:
                return seen, path + [action]
            if position not in seen:
                seen[position] = path + [action]
                queue.append((child, path + [action]))
    return seen, None


def observe(env):
    solve.solve(env)
    for top_phase in range(6):
        for main_phase in range(2):
            node = env.clone()
            for _ in range(top_phase):
                node.step(*TOP)
            for _ in range(main_phase):
                node.step(*MAIN)
            reached, win = movement_reach(node)
            grouped = {}
            for row, col in sorted(position for position in reached if position):
                grouped.setdefault(row, []).append(col)
            print(
                "REACH", top_phase, main_phase, len(reached),
                [(row, cols) for row, cols in grouped.items()],
                "WIN", win,
            )


arena.run_program("dc22", observe)
