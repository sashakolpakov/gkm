# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.


def repeat_action(env, action, count):
    """Take the same discrete action a fixed number of times."""
    for _ in range(count):
        if env.terminal():
            return
        env.step(action)


def diagnose_level(env):
    """Print a compact observational summary without changing the live state."""
    from perception import (
        action_deltas,
        color_counts,
        connected_components,
        frame_delta,
        object_candidates,
        replay,
    )

    print("L6_ACTIONS", list(env.actions))
    print("L6_COLORS", color_counts(env.frame()))
    objects = object_candidates(env.frame(), min_area=8)
    print("L6_OBJECTS", [
        (o["color"], o["bbox"], o["area"]) for o in objects
    ])
    deltas = action_deltas(env, actions=env.actions)
    print("L6_DELTAS", {
        a: (d["count"], d["bbox"]) for a, d in deltas.items()
    })
    for uses in range(5):
        node = replay(env, [5] * uses)
        ds = action_deltas(node, actions=(1, 2, 3, 4, 6))
        print("L6_MODE", uses, {
            a: (d["count"], d["bbox"]) for a, d in ds.items()
        })
    import numpy as np

    def tile_rows(node):
        frame = np.asarray(node.frame())[:63, :63]
        chars = {0: "0", 4: "4", 5: "5", 9: ".", 10: "A", 11: "B"}
        rows = []
        for r in range(0, 63, 3):
            row = []
            for c in range(0, 63, 3):
                vals, counts = np.unique(frame[r:r + 3, c:c + 3],
                                         return_counts=True)
                row.append(chars[int(vals[counts.argmax()])])
            rows.append("".join(row))
        return rows

    samples = {
        "initial": [],
        "piece1_right": [5, 4],
        "piece2_left": [5, 5, 3],
        "piece2_down": [5, 5, 2],
        "piece3_right": [5, 5, 5, 4],
    }
    for name, path in samples.items():
        print("L6_GRID", name)
        print("\n".join(tile_rows(replay(env, path))))

    for uses in (1, 2, 3):
        root = replay(env, [5] * uses)
        base_rows = tile_rows(root)
        for turns in range(4):
            node = replay(root, [6] * turns)
            rows = tile_rows(node)
            tile_changes = [
                (r, c, base_rows[r][c], rows[r][c])
                for r in range(21) for c in range(21)
                if base_rows[r][c] != rows[r][c]
            ]
            blobs = connected_components(
                node.frame(), colors=(4, 5, 10), min_area=4
            )
            sig = [(b.color, b.bbox, b.area) for b in blobs]
            print("L6_TURN", uses, turns, tile_changes, sig)
    from collections import deque

    target = np.asarray(env.frame())[:63, :63] == 11

    def ranked_states(prefix, colors, max_states, max_depth):
        root = replay(env, prefix)
        queue = deque([(root, ())])
        seen = set()
        ranked = []
        while queue and len(seen) < max_states:
            node, path = queue.popleft()
            frame = np.asarray(node.frame())[:63, :63]
            key = frame.tobytes()
            if key in seen:
                continue
            seen.add(key)
            colored = np.isin(frame, colors)
            on = int(np.count_nonzero(colored & target))
            off = int(np.count_nonzero(colored & ~target))
            ranked.append((on, -off, tuple(path)))
            if len(path) < max_depth:
                for action in (1, 2, 3, 4):
                    child = node.clone()
                    child.step(action)
                    queue.append((child, path + (action,)))
        return len(seen), sorted(ranked, reverse=True)

    ranked_paths = {}
    for label, prefix, colors, cap, depth in (
        ("mode0", [], (10,), 100, 22),
        ("mode1", [5], (4, 10), 100, 22),
        ("mode2", [5, 5], (5,), 500, 34),
        ("mode3", [5, 5, 5], (4, 5, 10), 500, 34),
    ):
        count, ranked = ranked_states(prefix, colors, cap, depth)
        ranked_paths[label] = [item[2] for item in ranked]
        print("L6_RANK", label, count, ranked[:16])

    base_level = env.levels_completed
    search_found = None
    mode0_path = [2] * 9
    for mode2_path in ([2] + [3] * 10, [2] + [3] * 15):
        tested = 0
        for mode1_path in ranked_paths["mode1"]:
            prefix_search = (
                mode0_path + [5] + list(mode1_path) + [5]
                + mode2_path + [5]
            )
            root = replay(env, prefix_search)
            queue = deque([(root, ())])
            seen = set()
            while queue and len(seen) < 400:
                node, path = queue.popleft()
                key = np.asarray(node.frame())[:63, :63].tobytes()
                if key in seen:
                    continue
                seen.add(key)
                if node.levels_completed > base_level:
                    search_found = prefix_search + list(path)
                    break
                for action in (1, 2, 3, 4):
                    child = node.clone()
                    child.step(action)
                    queue.append((child, path + (action,)))
            tested += len(seen)
            if search_found is not None:
                break
        print("L6_PRODUCT", tuple(mode2_path), tested, bool(search_found))
        if search_found is not None:
            break
    print("L6_PRODUCT_FOUND", search_found)
    return

    prefix = ([5] + [4] * 5 + [5] + [2] * 4 + [3] * 15 + [5])
    for path in ranked_paths["mode3"]:
        trial = replay(env, prefix + list(path) + [5] + [2] * 22)
        if trial.levels_completed > base_level:
            print("L6_FOUND", prefix + list(path) + [5] + [2] * 22)
            break
    else:
        print("L6_FOUND", None)

    found = None
    for down, left in ():
        mode3 = replay(
            env,
            [5] + [4] * 5 + [5] + [2] * down + [3] * left + [5],
        )
        for path in ranked_paths["mode3"]:
            trial = replay(mode3, path)
            trial.step(5)
            for _ in range(22):
                if trial.levels_completed > base_level:
                    found = (
                        [5] + [4] * 5 + [5] + [2] * down
                        + [3] * left + [5] + list(path) + [5]
                    )
                    break
                trial.step(2)
            if found is not None:
                break
        print("L6_EXHAUST", down, left, bool(found))
        if found is not None:
            break
    print("L6_EXHAUST_FOUND", found)

    for label, path in (
        ("mode1_placed", [5] + [4] * 5),
        ("mode2_placed", [5] + [4] * 5 + [5] + [2] * 4 + [3] * 15),
        ("mode3_placed", prefix + list(ranked_paths["mode3"][0])),
        ("scanner_staged", prefix + list(ranked_paths["mode3"][0]) + [5]),
    ):
        node = replay(env, path)
        before = node.frame()
        child = replay(node, [6])
        delta = frame_delta(before, child.frame())
        print("L6_SIX", label, child.levels_completed,
              delta["count"], delta["bbox"])
