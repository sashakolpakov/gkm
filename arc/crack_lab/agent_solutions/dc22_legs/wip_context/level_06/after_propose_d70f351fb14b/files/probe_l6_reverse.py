"""Find legal occupied handoffs for the reverse bridge transfer."""
from collections import deque
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve


TOP = (6, 54, 6)
TO_BRIDGE = [3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 2, 3]


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


def enter_left(env):
    node = env.clone()
    for _ in range(4):
        node.step(*TOP)
    for action in TO_BRIDGE:
        node.step(action)
    node.step(*TOP)
    node.step(1)
    node.step(3)
    node.step(*TOP)
    for _ in range(6):
        node.step(1)
    return node


def legal_handoffs(node):
    reached = movement_reach(node)
    legal = []
    for target, path in sorted(reached.items()):
        if target is None or target[0] < 54:
            continue
        candidate = node.clone()
        for action in path:
            candidate.step(action)
        before = perception.arr(candidate.frame())[54:60, 6:22].copy()
        candidate.step(*TOP)
        after = perception.arr(candidate.frame())[54:60, 6:22]
        if not (before == after).all():
            legal.append((target, path))
    return reached, legal


def observe(env):
    solve.solve(env)
    left = enter_left(env)
    reached, legal = legal_handoffs(left)
    print("PHASE0_REACH", sorted(reached))
    print("PHASE0_TO1", legal)
    phase_one = left.clone()
    selected_path = dict(legal)[(56, 8)]
    for action in selected_path:
        phase_one.step(action)
    phase_one.step(*TOP)
    reached_one, legal_one = legal_handoffs(phase_one)
    print("PHASE1_REACH", sorted(reached_one))
    print("PHASE1_TO2", legal_one)
    phase_two = phase_one.clone()
    selected_path = dict(legal_one)[(56, 8)]
    for action in selected_path:
        phase_two.step(action)
    phase_two.step(*TOP)
    reached_two, legal_two = legal_handoffs(phase_two)
    print("PHASE2_REACH", sorted(reached_two))
    print("PHASE2_TO3", legal_two)
    phase_three = phase_two.clone()
    selected_path = dict(legal_two)[(56, 10)]
    for action in selected_path:
        phase_three.step(action)
    phase_three.step(*TOP)
    reached_three, legal_three = legal_handoffs(phase_three)
    print("PHASE3_REACH", sorted(reached_three))
    print("PHASE3_TO4", legal_three)
    phase_four = phase_three.clone()
    selected_path = dict(legal_three)[(58, 10)]
    for action in selected_path:
        phase_four.step(action)
    phase_four.step(*TOP)
    reached_four = movement_reach(phase_four)
    print("PHASE4_REACH", sorted(reached_four))
    print("TO_REMOTE_PAD", reached_four.get((48, 18)))


arena.run_program("dc22", observe)
