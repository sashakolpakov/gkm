import importlib.util
import os
import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
import gkm_legs as G

from perception import (
    action_deltas,
    bounded_bfs,
    bounded_replay_bfs,
    color_counts,
    connected_components,
    frame_delta,
)


taint_reason = G._workspace_taint_reason(os.getcwd())
if taint_reason:
    raise SystemExit(f"TAINTED WORKSPACE: {taint_reason}")

spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def summarize(env):
    print("LEVEL", env.levels_completed + 1, "ACTIONS", env.actions)
    print("COUNTS", color_counts(env.frame()))
    blobs = connected_components(env.frame(), min_area=3)
    print("BLOBS")
    for blob in blobs:
        if blob.color != 2 or blob.area < 12:
            print(blob.color, blob.bbox, blob.area)
    print("LEFT_TILE_MAP")
    frame = env.frame()
    for r in range(0, 62, 2):
        print(f"{r // 2:02d}", "".join(f"{int(frame[r, c]):X}" for c in range(0, 40, 2)))
    print("KEY_DELTAS")
    for action, delta in action_deltas(env, env.actions).items():
        print(action, delta)


def run(env):
    solver.solve(env)
    summarize(env)

    base = env.frame()
    # Probe the centers of compact non-background components as likely controls.
    points = []
    for blob in connected_components(base, min_area=3):
        r0, c0, r1, c1 = blob.bbox
        if blob.area <= 40:
            points.append(((c0 + c1) // 2, (r0 + r1) // 2))
    print("COORD_DELTAS")
    for x, y in sorted(set(points)):
        clone = env.clone()
        clone.step(6, x, y)
        delta = frame_delta(base, clone.frame())
        if delta["count"]:
            print((x, y), {k: v for k, v in delta.items() if k != "samples"})

    active = env.clone()
    active.step(6, 56, 8)
    print("CONTROL_56_8", frame_delta(base, active.frame()))
    print("AFTER_CONTROL_KEY_DELTAS")
    for action, delta in action_deltas(active, active.actions).items():
        print(action, delta)

    def avatar_key(node):
        frame = node.frame()
        cells = tuple(
            (int(r), int(c))
            for r, c in zip(*((frame == 14).nonzero()))
            if int(c) < 40
        )
        return cells

    path = bounded_bfs(
        env,
        lambda node, _: node.levels_completed > 5,
        actions=(1, 2, 3, 4),
        key_fn=avatar_key,
        max_states=1200,
        max_depth=80,
    )
    print("MOVE_ONLY_WIN", path)

    def reachable(root, limit=500):
        start = avatar_key(root)
        queue = deque([(root.clone(), [])])
        seen = {start}
        won = None
        while queue and len(seen) < limit:
            node, node_path = queue.popleft()
            for action in (1, 2, 3, 4):
                child = node.clone()
                child.step(action)
                child_path = node_path + [action]
                if child.levels_completed > 5:
                    return seen, child_path
                key = avatar_key(child)
                if key not in seen:
                    seen.add(key)
                    queue.append((child, child_path))
        return seen, won

    print("CONTROL_PHASES")
    phased = env.clone()
    for phase in range(0):
        positions, win = reachable(phased)
        anchors = sorted((cells[0][0] // 2, cells[0][1] // 2) for cells in positions if cells)
        one_blobs = [(b.bbox, b.area) for b in connected_components(phased.frame(), colors=(1,), min_area=1)]
        print(phase, "assembly", one_blobs, "reachable", anchors, "win", win)
        phased.step(6, 56, 8)

    def compact_key(node):
        frame = node.frame()
        return avatar_key(node), frame[54:60, 6:20].tobytes()

    transfer_path = [
        3, 3, 3, 3, 3,
        (6, 56, 8), (6, 56, 8), (6, 56, 8), (6, 56, 8),
        2, 2, 3, 3, 3, 2, 3, (6, 56, 8), 1, (6, 56, 8), 1, 1,
    ]
    print("LEFT_ISLAND_TRANSFER", transfer_path)
    if transfer_path:
        reached = env.clone()
        for action in transfer_path:
            reached.step(*action) if isinstance(action, tuple) else reached.step(action)
        print("REACHED_BLOBS_RIGHT")
        for blob in connected_components(reached.frame(), min_area=3):
            if blob.bbox[1] >= 40 and blob.area <= 100:
                print(blob.color, blob.bbox, blob.area)
        print("REACHED_COORD_DELTAS")
        reached_base = reached.frame()
        for blob in connected_components(reached_base, min_area=3):
            r0, c0, r1, c1 = blob.bbox
            if c0 < 40 or blob.area > 100:
                continue
            point = ((c0 + c1) // 2, (r0 + r1) // 2)
            child = reached.clone()
            child.step(6, *point)
            delta = frame_delta(reached_base, child.frame())
            if delta["count"]:
                print(point, delta)

        print("SECOND_CONTROL_PHASES")
        second_phased = reached.clone()
        for phase in range(0):
            positions, win = reachable(second_phased)
            anchors = sorted(
                (cells[0][0] // 2, cells[0][1] // 2)
                for cells in positions
                if cells
            )
            print(
                phase,
                "bounds",
                (min(anchors), max(anchors)),
                "count",
                len(anchors),
                "win",
                win,
            )
            second_phased.step(6, 51, 25)

        second_root = reached.clone()
        second_root.step(6, 51, 25)
        second_path = bounded_bfs(
            second_root,
            lambda node, _: (
                bool(avatar_key(node))
                and (
                    avatar_key(node)[0][0] // 2,
                    avatar_key(node)[0][1] // 2,
                ) == (9, 4)
            ),
            actions=(1, 2, 3, 4),
            key_fn=avatar_key,
            max_states=200,
            max_depth=50,
        )
        print("SECOND_TRANSFER", [(6, 51, 25)] + second_path)
        reached2 = second_root.clone()
        for action in second_path:
            reached2.step(action)
        print("SECOND_REACHED_COORD_DELTAS")
        base2 = reached2.frame()
        for blob in connected_components(base2, min_area=3):
            r0, c0, r1, c1 = blob.bbox
            if c0 < 40 or blob.area > 100:
                continue
            point = ((c0 + c1) // 2, (r0 + r1) // 2)
            child = reached2.clone()
            child.step(6, *point)
            delta = frame_delta(base2, child.frame())
            if delta["count"]:
                print(point, {k: v for k, v in delta.items() if k != "samples"})


A.run_program("dc22", run)
