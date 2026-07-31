"""Contextual control probes with the avatar staged on moving structures."""
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
    queue = deque([(root.clone(), [])])
    seen = {avatar_position(root): []}
    while queue:
        node, path = queue.popleft()
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            position = avatar_position(child)
            if position not in seen:
                seen[position] = path + [action]
                queue.append((child, path + [action]))
    return seen


def grouped(reached):
    rows = {}
    for row, col in sorted(position for position in reached if position):
        rows.setdefault(row, []).append(col)
    return [(row, cols) for row, cols in rows.items()]


def observe(env):
    solve.solve(env)
    node = env.clone()
    for _ in range(4):
        node.step(*TOP)
    first_reach = movement_reach(node)
    target = (58, 10)
    path = first_reach[target]
    for action in path:
        node.step(action)
    print("STAGED", avatar_position(node), path)
    for phase in (5, 0, 1):
        before = avatar_position(node)
        node.step(*TOP)
        reached = movement_reach(node)
        print(
            "AFTER_TOP", phase, before, avatar_position(node),
            len(reached), grouped(reached),
        )
    phase_five = env.clone()
    for _ in range(5):
        phase_five.step(*TOP)
    fifth_reach = movement_reach(phase_five)
    for target, target_path in sorted(fifth_reach.items()):
        if target is None or target[0] < 54:
            continue
        candidate = phase_five.clone()
        for action in target_path:
            candidate.step(action)
        before_bridge = perception.arr(candidate.frame())[54:60, 6:22].copy()
        candidate.step(*TOP)
        after_bridge = perception.arr(candidate.frame())[54:60, 6:22]
        advanced = not (before_bridge == after_bridge).all()
        print("PHASE5_CLICK", target, advanced, avatar_position(candidate))
    transfer = env.clone()
    for _ in range(4):
        transfer.step(*TOP)
    for action in first_reach[(58, 10)]:
        transfer.step(action)
    transfer.step(*TOP)
    transfer.step(1)
    transfer.step(3)
    transfer.step(*TOP)
    transferred_reach = movement_reach(transfer)
    print(
        "TRANSFERRED", avatar_position(transfer), len(transferred_reach),
        grouped(transferred_reach),
    )
    for _ in range(6):
        transfer.step(1)
    transfer.step(*MAIN)
    upper_reach = movement_reach(transfer)
    print(
        "MAIN_VERTICAL", avatar_position(transfer), len(upper_reach),
        grouped(upper_reach),
    )
    if (16, 8) in upper_reach:
        print("TO_UPPER", upper_reach[(16, 8)])


arena.run_program("dc22", observe)
