"""Replay-based BFS from the verified upper-middle checkpoint."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena

import perception
import solve


TOP = (6, 54, 6)
MAIN = (6, 50, 26)
NEW = (6, 50, 46)
TO_BRIDGE = [3, 3, 3, 3, 3, 2, 2, 3, 3, 3, 2, 3]


def enter_upper(env):
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
    for _ in range(14):
        node.step(1)
    return node


def observe(env):
    solve.solve(env)
    upper = enter_upper(env)
    base_level = upper.levels_completed
    path = perception.bounded_replay_bfs(
        upper,
        perception.level_goal(base_level),
        lambda node: (1, 2, 3, 4, MAIN, NEW),
        key_fn=symbolic_key,
        max_states=700,
        max_depth=45,
    )
    print("UPPER_BFS", path)


def symbolic_key(node):
    frame = perception.arr(node.frame())
    avatar_pixels = perception.connected_components(
        frame, colors=(14,), min_area=2
    )
    avatar = next(
        (blob.top_left for blob in avatar_pixels if blob.bbox[1] < 32),
        None,
    )
    main_phase = int(frame[4, 4])
    selector = tuple(int(value) for value in frame[48:50, 18:20].flat)
    return avatar, main_phase, selector


arena.run_program("dc22", observe)
