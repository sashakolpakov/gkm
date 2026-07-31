"""Compare the central-rotator action at every ring placement."""
from collections import deque
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve
from probe_l6_right import MAIN, avatar_position, enter_right


CENTER = (58, 34)
MOVES = (
    (1, (6, 50, 34)), (2, (6, 50, 40)),
    (3, (6, 46, 36)), (4, (6, 54, 36)),
)
TO_CENTER = {
    (56, 34): 2, (60, 34): 1, (58, 32): 4, (58, 36): 3,
}


def ring_key(env):
    return perception.arr(env.frame())[6:42, 6:34].tobytes()


def ring_label(env):
    return tuple(
        (blob.bbox, blob.area)
        for blob in perception.connected_components(
            env.frame(), colors=(8,), min_area=20
        )
        if blob.bbox[1] < 32
    )


def placements(root):
    queue = deque([root.clone()])
    seen = {ring_key(root)}
    out = [root.clone()]
    while queue:
        node = queue.popleft()
        position = avatar_position(node)
        for movement, control in MOVES:
            child = node.clone()
            if position != CENTER:
                child.step(TO_CENTER[position])
            child.step(movement)
            child.step(*control)
            if avatar_position(child) != CENTER:
                child.step(TO_CENTER[avatar_position(child)])
            key = ring_key(child)
            if key in seen:
                continue
            seen.add(key)
            out.append(child.clone())
            queue.append(child)
    return out


def observe(env):
    solve.solve(env)
    nodes = placements(enter_right(env, 3))
    for index, placement in enumerate(nodes):
        staged = placement.clone()
        staged.step(1)
        before = perception.arr(staged.frame()).copy()
        once = staged.clone()
        once.step(*MAIN)
        delta = perception.frame_delta(before, once.frame())
        twice = once.clone()
        twice.step(*MAIN)
        after = perception.arr(twice.frame()).copy()
        reversible = bool((before[:63] == after[:63]).all())
        residual = perception.frame_delta(before[:63], after[:63])
        print(
            "MAIN_RING_CONTEXT", index, ring_label(staged),
            "DELTA", delta["count"], delta["bbox"],
            "RING_AFTER", ring_label(once),
            "REVERSIBLE", reversible,
            "RESIDUAL", residual["count"], residual["bbox"],
        )
    docked = nodes[13].clone()
    docked.step(3)
    docked.step(*MAIN)
    docked.step(4)
    docked.step(4)
    docked.step(6, 54, 36)
    reference = nodes[11].clone()
    reference.step(3)
    reference.step(*MAIN)
    reference.step(4)
    reference.step(4)
    delta = perception.frame_delta(
        perception.arr(reference.frame())[:44, :40],
        perception.arr(docked.frame())[:44, :40],
    )
    print("MAIN_DOCK_DEPARTURE_DELTA", delta)


arena.run_program("dc22", observe)
