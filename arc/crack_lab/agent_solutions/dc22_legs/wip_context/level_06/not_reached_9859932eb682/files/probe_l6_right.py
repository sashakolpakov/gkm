"""Reachability and affordances in the two right-hand destinations."""
from collections import deque
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve


TOP = (6, 54, 6)
MAIN = (6, 50, 26)
SELECTOR = (6, 50, 46)
TO_BRIDGE = [3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 2, 3]
TO_REMOTE_PAD = [4, 1, 4, 1, 4, 4, 1, 1, 1]


def avatar_position(env):
    blobs = perception.connected_components(
        env.frame(), colors=(14,), min_area=4
    )
    avatars = [blob for blob in blobs if blob.area == 4]
    return avatars[0].top_left if avatars else None


def enter_glyph(env):
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
    node.step(*MAIN)
    for _ in range(13):
        node.step(1)
    return node


def enter_right(env, selector_steps):
    node = enter_glyph(env)
    for _ in range(selector_steps):
        node.step(*SELECTOR)
    for _ in range(13):
        node.step(2)
    for _ in range(6):
        node.step(2)
    node.step(*TOP)
    node.step(*TOP)
    node.step(4)
    node.step(*TOP)
    node.step(2)
    node.step(*TOP)
    for action in TO_REMOTE_PAD:
        node.step(action)
    node.step(*MAIN)
    return node


def movement_reach(root):
    queue = deque([(root.clone(), [])])
    seen = {avatar_position(root): []}
    win = None
    while queue:
        node, path = queue.popleft()
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            child_path = path + [action]
            if child.levels_completed > root.levels_completed:
                win = child_path
                return seen, win
            position = avatar_position(child)
            if position not in seen:
                seen[position] = child_path
                queue.append((child, child_path))
    return seen, win


def scan_controls(node):
    frame = perception.arr(node.frame()).copy()
    found = []
    for y in range(2, 64, 4):
        for x in range(2, 64, 4):
            clone = node.clone()
            clone.step(6, x, y)
            delta = perception.frame_delta(frame, clone.frame())
            samples = [
                sample for sample in delta["samples"] if sample[0] < 63
            ]
            if samples:
                found.append(((x, y), delta["count"], delta["bbox"]))
    return found


def observe(env):
    solve.solve(env)
    selected_states = tuple(int(value) for value in sys.argv[1:]) or (0, 3)
    for selector_steps in selected_states:
        right = enter_right(env, selector_steps)
        region = (
            perception.arr(right.frame())[48:54, 32:38]
            if selector_steps == 0
            else perception.arr(right.frame())[56:62, 32:38]
        )
        print("RIGHT_TILE", selector_steps, region.tolist())
        reached, win = movement_reach(right)
        print(
            "RIGHT", selector_steps, avatar_position(right),
            sorted(position for position in reached if position is not None),
            "VANISHED", None in reached, "WIN", win,
        )
        for target, path in sorted(
                (item for item in reached.items() if item[0] is not None)):
            branch = right.clone()
            for action in path:
                branch.step(action)
            before_phase = int(perception.arr(branch.frame())[4, 4])
            branch.step(*MAIN)
            after_phase = int(perception.arr(branch.frame())[4, 4])
            print(
                "RIGHT_MAIN", selector_steps, target,
                before_phase != after_phase, avatar_position(branch),
                branch.levels_completed,
            )
            for selector_offset in range(4):
                selector_branch = right.clone()
                for action in path:
                    selector_branch.step(action)
                for _ in range(selector_offset):
                    selector_branch.step(*SELECTOR)
                selector_branch.step(*MAIN)
                destination = avatar_position(selector_branch)
                if (
                    destination != target
                    or selector_branch.levels_completed
                    > right.levels_completed
                ):
                    print(
                        "RIGHT_SELECTOR_BRANCH", selector_steps, target,
                        selector_offset, destination,
                        selector_branch.levels_completed,
                    )
        safe_target = (52, 34) if selector_steps == 0 else (56, 34)
        rotated = right.clone()
        for action in reached[safe_target]:
            rotated.step(action)
        rotated.step(*MAIN)
        rotated_reach, rotated_win = movement_reach(rotated)
        print(
            "RIGHT_ROTATED", selector_steps, avatar_position(rotated),
            sorted(
                position for position in rotated_reach
                if position is not None
            ),
            "VANISHED", None in rotated_reach, "WIN", rotated_win,
        )
        if selector_steps == 0:
            for outgoing_target in ((48, 34), (48, 36)):
                outgoing = right.clone()
                for action in reached[outgoing_target]:
                    outgoing.step(action)
                outgoing.step(*SELECTOR)
                for main_click in range(1, 5):
                    outgoing.step(*MAIN)
                    print(
                        "RIGHT_OUTGOING", outgoing_target, main_click,
                        int(perception.arr(outgoing.frame())[4, 4]),
                        avatar_position(outgoing), outgoing.levels_completed,
                        [
                            (blob.bbox, blob.area)
                            for blob in perception.connected_components(
                                outgoing.frame(), colors=(14,), min_area=1
                            )
                        ],
                    )
            for outgoing_target in ((48, 34), (48, 36)):
                chain = right.clone()
                for action in reached[outgoing_target]:
                    chain.step(action)
                if outgoing_target == (48, 34):
                    return_path = (3, 2, 2)
                else:
                    return_path = (3, 3, 2, 2)
                for action in return_path:
                    chain.step(action)
                chain.step(*MAIN)
                hub_position = avatar_position(chain)
                chain.step(*SELECTOR)
                chain.step(*MAIN)
                print(
                    "RIGHT_CHAIN1", outgoing_target, hub_position,
                    avatar_position(chain), chain.levels_completed,
                )
            glyph_context = right.clone()
            for action in reached[(48, 34)]:
                glyph_context.step(action)
            print("RIGHT_GLYPH_CONTROLS", scan_controls(glyph_context))
            glyph_context_right = right.clone()
            for action in reached[(48, 36)]:
                glyph_context_right.step(action)
            print(
                "RIGHT_GLYPH_RIGHT_CONTROLS",
                scan_controls(glyph_context_right),
            )
        if selector_steps == 3:
            goal_context = right.clone()
            for action in reached[(56, 34)]:
                goal_context.step(action)
            print("RIGHT_GOAL_CONTROLS", scan_controls(goal_context))
            final_control = (6, 50, 34)
            for click in range(7):
                central = [
                    (blob.bbox, blob.area)
                    for blob in perception.connected_components(
                        goal_context.frame(), colors=(8,), min_area=2
                    )
                    if blob.bbox[1] < 32
                ]
                print(
                    "FINAL_CYCLE", click, central,
                    goal_context.levels_completed,
                )
                goal_context.step(*final_control)
            central_chain = right.clone()
            for action in reached[(56, 34)]:
                central_chain.step(action)
            central_chain.step(*final_control)
            central_chain.step(2)
            central_chain.step(*MAIN)
            hub_position = avatar_position(central_chain)
            central_chain.step(*SELECTOR)
            central_chain.step(*SELECTOR)
            central_chain.step(*MAIN)
            print(
                "RIGHT_CHAIN_CENTRAL", hub_position,
                avatar_position(central_chain),
                central_chain.levels_completed,
                [
                    (blob.bbox, blob.area)
                    for blob in perception.connected_components(
                        central_chain.frame(), colors=(14,), min_area=1
                    )
                ],
            )
            synchronized = right.clone()
            for action in reached[(56, 34)]:
                synchronized.step(action)
            for phase in range(1, 9):
                synchronized.step(*final_control)
                central = [
                    blob.bbox
                    for blob in perception.connected_components(
                        synchronized.frame(), colors=(8,), min_area=20
                    )
                    if blob.bbox[1] < 32
                ]
                print(
                    "FINAL_SYNC", phase, central,
                    avatar_position(synchronized),
                    synchronized.levels_completed,
                )
                synchronized.step(2 if phase % 2 else 1)
            routed = right.clone()
            for action in reached[(56, 34)]:
                routed.step(action)
            for phase in range(1, 9):
                routed.step(*final_control)
                central = [
                    blob.bbox
                    for blob in perception.connected_components(
                        routed.frame(), colors=(8,), min_area=20
                    )
                    if blob.bbox[1] < 32
                ]
                print(
                    "FINAL_MAIN_SYNC", phase, central,
                    int(perception.arr(routed.frame())[4, 4]),
                    avatar_position(routed), routed.levels_completed,
                )
                routed.step(*MAIN)
        print("RIGHT_CONTROLS", selector_steps, scan_controls(right))


if __name__ == "__main__":
    arena.run_program("dc22", observe)
