"""Compact observational probes for the current level-3 frontier."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A

from perception import action_deltas, arr, bounded_bfs, color_counts, connected_components, frame_delta
from players import play_level_1, play_level_2


UP, DOWN, RETRACT, EXTEND, USE = 1, 2, 3, 4, 6


def compact_blobs(frame):
    return [
        (b.color, b.bbox, b.area)
        for b in connected_components(frame, min_area=4)
        if b.color != 1
    ]


def token_state(env):
    return {
        color: [(b.bbox, b.area) for b in connected_components(env.frame(), [color], min_area=4)]
        for color in (8, 9, 12, 14)
    }


def rig_state(env):
    return [
        (b.color, b.bbox, b.area)
        for b in connected_components(env.frame(), [0, 6], min_area=4)
        if b.bbox[0] < 53
    ]


def row_runs(env, row):
    values = list(map(int, env.frame()[row]))
    out = []
    start = 0
    for col in range(1, len(values) + 1):
        if col == len(values) or values[col] != values[start]:
            out.append((start, col - 1, values[start]))
            start = col
    return out


def salient_row(env, row):
    return [run for run in row_runs(env, row) if run[2] not in (4, 5)]


def run_segments(root, label, segments):
    node = root.clone()
    print("CASE", label, "START", token_state(node), rig_state(node))
    for name, action, count in segments:
        for _ in range(count):
            before = node.frame()
            node.step(action)
            delta = frame_delta(before, node.frame())
            print("STEP", name, delta["count"], delta["bbox"], rig_state(node))
        print(name, "L", node.levels_completed, token_state(node), rig_state(node),
              "ROW", row_runs(node, 20) if label == "short_reach_order" else ())


def inspect(env):
    initial = env.clone()
    print("LEVEL1_START", token_state(initial), rig_state(initial))
    for action, count in ((UP, 3), (EXTEND, 4), (RETRACT, 4)):
        for _ in range(count):
            initial.step(action)
        print("LEVEL1_PHASE", action, count, initial.levels_completed, token_state(initial), rig_state(initial))
    play_level_1(env)
    play_level_2(env)
    print("LEVEL", env.levels_completed, "ACTIONS", env.actions)
    print("COUNTS", color_counts(env.frame()))
    print("BLOBS", compact_blobs(env.frame()))
    print(
        "DELTAS",
        {
            action: (delta["count"], delta["bbox"])
            for action, delta in action_deltas(env, env.actions).items()
        },
    )
    for label, x, y in (
        ("avatar", 7, 44),
        ("field14", 31, 8),
        ("field9", 31, 14),
        ("field8", 31, 20),
        ("field12", 31, 26),
        ("request8", 25, 58),
        ("request12", 31, 58),
        ("request9", 37, 58),
        ("request14", 43, 58),
    ):
        node = env.clone()
        before = node.frame()
        node.step(6, x, y)
        delta = frame_delta(before, node.frame())
        print("CLICK", label, (x, y), delta["count"], delta["bbox"],
              node.levels_completed, token_state(node))
    idle = env.clone()
    for turn in range(1, 17):
        before = idle.frame()
        idle.step(DOWN)
        delta = frame_delta(before, idle.frame())
        print("IDLE", turn, delta["count"], delta["bbox"],
              salient_row(idle, 58), idle.levels_completed)
    run_segments(
        env,
        "collect_8_then_12",
        (
            ("up4", UP, 4),
            ("extend4", EXTEND, 4),
            ("retract4", RETRACT, 4),
            ("down1", DOWN, 1),
            ("extend4b", EXTEND, 4),
            ("retract4b", RETRACT, 4),
        ),
    )
    run_segments(
        env,
        "short_reach_order",
        (
            ("up4", UP, 4),
            ("extend2", EXTEND, 2),
            ("use", USE, 1),
            ("retract2", RETRACT, 2),
            ("down1", DOWN, 1),
            ("extend2b", EXTEND, 2),
            ("retract1b", RETRACT, 1),
            ("up2", UP, 2),
        ),
    )
    for color, lanes in ((14, 6), (9, 5), (8, 4), (12, 3)):
        run_segments(
            env,
            f"first_{color}",
            (("approach", UP, lanes), ("reach", EXTEND, 2), ("pull", RETRACT, 2)),
        )
    run_segments(
        env,
        "reset_then_hook_8",
        (
            ("reset", RETRACT, 1),
            ("up4", UP, 4),
            ("reach4", EXTEND, 4),
            ("pull4", RETRACT, 4),
        ),
    )
    run_segments(
        env,
        "wall_hook_8",
        (("up4", UP, 4), ("wall", EXTEND, 6), ("pull", RETRACT, 6)),
    )
    search_root = env.clone()
    for _ in range(4):
        search_root.step(UP)

    def moved_field_8(node, _path):
        return any(
            blob.bbox[0] < 53 and blob.bbox != (19, 30, 22, 33)
            for blob in connected_components(node.frame(), [8], min_area=4)
        )

    path = bounded_bfs(
        search_root,
        moved_field_8,
        actions=(UP, DOWN, RETRACT, EXTEND),
        key_fn=lambda node: arr(node.frame())[:53].tobytes(),
        max_states=5000,
        max_depth=10,
    )
    print("BFS_MOVED_8", path)
    moved = search_root.clone()
    for action in path or ():
        moved.step(action)
    print("BFS_MOVED_STATE", moved.levels_completed, token_state(moved))
    win_path = bounded_bfs(
        env,
        lambda node, _path: node.levels_completed > 2,
        actions=(UP, DOWN, RETRACT, EXTEND),
        key_fn=lambda node: arr(node.frame())[:53].tobytes(),
        max_states=20000,
        max_depth=35,
    )
    print("BFS_WIN", win_path)


if __name__ == "__main__":
    levels, path, err = A.run_program("sk48", inspect)
    print("END", levels, len(path), err)
