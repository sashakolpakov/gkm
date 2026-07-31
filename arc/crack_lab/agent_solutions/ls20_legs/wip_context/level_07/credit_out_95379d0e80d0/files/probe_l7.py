import sys
from collections import deque
from itertools import permutations

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import players
from perception import color_counts, connected_components, frame_delta


def enter_level_7(env):
    for level in range(1, 7):
        getattr(players, f"play_level_{level}")(env)


def tile_map(frame):
    data = np.asarray(frame)
    symbols = []
    for row in range(0, 40, 5):
        line = []
        for col in range(4, 44, 5):
            tile = data[row : row + 5, col : col + 5]
            values = set(int(value) for value in np.unique(tile))
            if 9 in values or 12 in values:
                symbol = "A"
            elif 11 in values:
                symbol = "o"
            elif 1 in values:
                symbol = "p"
            else:
                counts = {
                    color: int(np.count_nonzero(tile == color))
                    for color in (3, 4, 5)
                }
                symbol = {3: "#", 4: ".", 5: " "}[max(counts, key=counts.get)]
            line.append(symbol)
        symbols.append("".join(line))
    return tuple(symbols)


def avatar_position(frame):
    avatars = [
        blob
        for blob in connected_components(frame, colors=(9,), min_area=15)
        if blob.area == 15 and blob.bbox[0] < 55
    ]
    return avatars[0].bbox[:2] if len(avatars) == 1 else None


def meter(frame):
    data = np.asarray(frame)
    return int(np.count_nonzero(data[61:63] == 11) // 4)


def world_objects(frame):
    return [
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, colors=(1, 11), min_area=1)
        if blob.bbox[0] < 50
    ]


def world_features(frame):
    avatar = avatar_position(frame)
    found = []
    for blob in connected_components(frame, min_area=1):
        if blob.bbox[0] >= 50 or blob.color in (3, 4, 5):
            continue
        if (
            avatar is not None
            and blob.color in (9, 12)
            and avatar[0] - 2 <= blob.bbox[0] <= avatar[0] + 2
            and avatar[1] <= blob.bbox[1] <= avatar[1] + 4
        ):
            continue
        if blob.color in (1, 11) and blob.area < (5 if blob.color == 1 else 8):
            continue
        found.append((blob.color, blob.bbox, blob.area))
    return found


def secondary_components(frame):
    main = avatar_position(frame)
    found = []
    for blob in connected_components(frame, colors=(0, 8, 9, 12, 14), min_area=1):
        if blob.bbox[0] >= 55:
            continue
        if (
            main is not None
            and blob.color in (9, 12)
            and main[0] - 2 <= blob.bbox[0] <= main[0] + 2
            and main[1] <= blob.bbox[1] <= main[1] + 4
        ):
            continue
        found.append((blob.color, blob.bbox, blob.area))
    return found


def tile_patterns(frame):
    data = np.asarray(frame)
    main = avatar_position(frame)
    patterns = []
    for row in range(1, 54, 5):
        for col in range(0, 62, 5):
            tile = data[row : row + 3, col : col + 3]
            if tile.shape != (3, 3):
                continue
            if main is not None:
                main_top = main[0] - 2
                main_left = main[1]
                if (
                    row < main_top + 5
                    and row + 3 > main_top
                    and col < main_left + 5
                    and col + 3 > main_left
                ):
                    continue
            special = int(np.count_nonzero(~np.isin(tile, (3, 4, 5))))
            if special >= 5:
                pattern = tuple(
                    "".join("." if int(value) == 5 else f"{int(value):X}" for value in line)
                    for line in tile
                )
                patterns.append(((row, col), pattern))
    return patterns


def inspect_multiagent(root):
    base = root.clone()
    for action in (3, 3, 2, 2):
        base.step(action)
    print("multi base", avatar_position(base.frame()), secondary_components(base.frame()))
    for action in base.actions:
        child = base.clone()
        child.step(int(action))
        print(
            "multi action",
            int(action),
            avatar_position(child.frame()),
            secondary_components(child.frame()),
            hud_state(child.frame()),
        )


def symbolic_crop(frame, rows, cols):
    data = np.asarray(frame)[rows, cols]
    return tuple(
        "".join("." if int(value) == 5 else f"{int(value):X}" for value in row)
        for row in data
    )


def inspect_lower(root):
    clone = root.clone()
    for action in (3, 3, 2, 2):
        clone.step(action)
    print("lower_crop", "/".join(symbolic_crop(clone.frame(), slice(39, 46), slice(8, 25))))
    print("hud", "/".join(symbolic_crop(clone.frame(), slice(55, 61), slice(3, 9))))
    trace_clone = root.clone()
    states = []
    for action in (3, 3, 2, 2, 2, 2, 2, 4, 2):
        trace_clone.step(action)
        states.append(
            (
                avatar_position(trace_clone.frame()),
                meter(trace_clone.frame()),
                symbolic_crop(trace_clone.frame(), slice(55, 61), slice(3, 9)),
                int(trace_clone.levels_completed),
            )
        )
    print("lower_trace", states)


def inspect_agents(root, moving=False):
    clone = root.clone()
    for action in (3, 3, 2, 2):
        clone.step(action)
    actions = (1, 2, 1, 2, 1, 2, 1, 2) if moving else (3,) * 8
    for turn in range(9):
        print(
            "agents",
            turn,
            avatar_position(clone.frame()),
            meter(clone.frame()),
            "/".join(symbolic_crop(clone.frame(), slice(39, 46), slice(8, 40))),
            "/".join(symbolic_crop(clone.frame(), slice(55, 61), slice(3, 9))),
        )
        if turn < len(actions):
            clone.step(actions[turn])


def candidate(root, name, actions):
    clone = root.clone()
    states = []
    for action in actions:
        if clone.levels_completed > 6:
            break
        clone.step(action)
        states.append(
            (
                compact_path((action,)),
                avatar_position(clone.frame()),
                meter(clone.frame()),
                symbolic_crop(clone.frame(), slice(55, 61), slice(3, 9)),
                int(clone.levels_completed),
            )
        )
    print("candidate", name, compact_path(actions), states[-4:])


def hud_state(frame):
    glyph = np.asarray(frame)[55:61:2, 3:9:2]
    return tuple(
        "".join("." if int(value) == 5 else f"{int(value):X}" for value in row)
        for row in glyph
    )


def cycle_probe(root):
    tests = {
        "color": ((3, 3, 2, 2, 2, 2, 2), (4, 3)),
        "color_vertical": ((3, 3, 2, 2, 2, 2, 2), (1, 2)),
        "shape": ((3, 3, 2, 2, 2, 2, 4, 4, 2), (3, 4)),
        "shape_vertical": ((3, 3, 2, 2, 2, 2, 4, 4, 2), (1, 2)),
    }
    for name, (prefix, cycle) in tests.items():
        clone = root.clone()
        for action in prefix:
            clone.step(action)
        states = [(1, avatar_position(clone.frame()), meter(clone.frame()), hud_state(clone.frame()))]
        for visit in range(2, 6):
            for action in cycle:
                clone.step(action)
            states.append(
                (visit, avatar_position(clone.frame()), meter(clone.frame()), hud_state(clone.frame()))
            )
        print("cycle", name, compact_path(prefix), compact_path(cycle), states)
    clone = root.clone()
    prefix = (3, 3, 2, 2, 2, 2, 2, 4, 2, 1, 4)
    for action in prefix:
        clone.step(action)
    states = [hud_state(clone.frame())]
    for _ in range(9):
        for action in (3, 4):
            clone.step(action)
        states.append(hud_state(clone.frame()))
    print("cycle_refilled_shape", states)


def inspect_target(root):
    paths = {
        "bottom_right": (1, 1, 4, 4, 2, 4, 2, 2, 4, 4, 4, 2),
        "right_lower": (1, 1, 4, 4, 2, 4, 2, 2, 4, 4, 4, 4),
    }
    for name, path in paths.items():
        clone = root.clone()
        for action in path:
            clone.step(action)
        target = np.asarray(clone.frame())[51:54, 55:58]
        hud = np.asarray(clone.frame())[55:61:2, 3:9:2]
        print(
            "target",
            name,
            compact_path(path),
            avatar_position(clone.frame()),
            hud_state(clone.frame()),
            symbolic_crop(clone.frame(), slice(51, 54), slice(55, 58)),
            "mismatch",
            int(np.count_nonzero(hud != target)),
            "near",
            "/".join(symbolic_crop(clone.frame(), slice(49, 55), slice(52, 61))),
        )


def inspect_right_edge(root):
    clone = root.clone()
    prefix = (1, 1, 4, 4, 2, 4, 2, 2, 4, 4, 4, 2, 2, 4, 3, 1, 1, 4)
    for action in prefix:
        clone.step(action)
    states = [(avatar_position(clone.frame()), meter(clone.frame()), hud_state(clone.frame()))]
    for _ in range(7):
        clone.step(1)
        states.append((avatar_position(clone.frame()), meter(clone.frame()), hud_state(clone.frame())))
    print("right_edge", compact_path(prefix), states)
    clone = root.clone()
    prefix = (1, 1, 4, 4, 2, 4, 2, 2, 4, 4, 4, 2, 2, 4, 3, 1, 1, 4, 1, 1, 1)
    for action in prefix:
        clone.step(action)
    states = []
    for visit in range(1, 7):
        clone.step(1)
        states.append((visit, meter(clone.frame()), hud_state(clone.frame())))
        if visit < 6:
            clone.step(2)
    print("right_edge_cycle", compact_path(prefix), states)


def state_candidate_path(color_index, shape_index):
    if color_index == 0:
        path = [3, 3] + [2] * 4 + [4, 2, 2]
    else:
        path = [3, 3] + [2] * 5
        for _ in range(color_index - 1):
            path += [4, 3]
        path += [4, 2]
    if shape_index == 0:
        path += [1, 1, 3] + [1] * 6
    else:
        path += [1, 4]
        for _ in range(shape_index - 1):
            path += [3, 4]
        path += [1, 3, 3] + [1] * 6
    path += [4] * 4 + [2, 4] + [2] * 2 + [4] * 3 + [2] * 2 + [4]
    return tuple(path)


def brute_states(root):
    results = []
    for color_index in range(4):
        for shape_index in range(6):
            clone = root.clone()
            path = state_candidate_path(color_index, shape_index)
            for action in path:
                if clone.levels_completed > 6:
                    break
                clone.step(action)
            results.append(
                (
                    color_index,
                    shape_index,
                    len(path),
                    int(clone.levels_completed),
                    avatar_position(clone.frame()),
                    meter(clone.frame()),
                    hud_state(clone.frame()),
                )
            )
    print("state_trials", results)


def brute_remote_states(root):
    excursion = (3, 1, 1, 4, 1, 1, 1, 1, 2, 2, 2, 2, 3, 2, 2, 4)
    results = []
    for color_index in range(4):
        for shape_index in range(6):
            clone = root.clone()
            path = state_candidate_path(color_index, shape_index) + excursion
            changes = []
            for step, action in enumerate(path, 1):
                if clone.levels_completed > 6:
                    break
                old_hud = hud_state(clone.frame())
                clone.step(action)
                new_hud = hud_state(clone.frame())
                if new_hud != old_hud:
                    changes.append((step, new_hud))
            results.append(
                (
                    color_index,
                    shape_index,
                    int(clone.levels_completed),
                    avatar_position(clone.frame()),
                    meter(clone.frame()),
                    hud_state(clone.frame()),
                    changes[-5:],
                )
            )
    print("remote_state_trials", results)


def brute_target_gates(root):
    excursion = (3, 1, 1, 4, 1, 1, 1, 1, 2, 2, 2, 2)
    tails = {
        "vertical": (2, 2),
        "corner": (3, 2, 4, 2),
    }
    results = []
    for color_index in range(4):
        for shape_index in range(6):
            for name, tail in tails.items():
                clone = root.clone()
                path = state_candidate_path(color_index, shape_index) + excursion + tail
                for action in path:
                    if clone.levels_completed > 6:
                        break
                    clone.step(action)
                results.append(
                    (
                        color_index,
                        shape_index,
                        name,
                        int(clone.levels_completed),
                        avatar_position(clone.frame()),
                        meter(clone.frame()),
                        hud_state(clone.frame()),
                    )
                )
    print("gate_trials", results)


def phase_candidate(root):
    base = state_candidate_path(1, 4)
    trigger = (3, 1, 1, 4, 1, 1, 1, 1)
    finish = (3,) + (2,) * 6 + (4,)
    results = []
    for delay in (0, 2, 4):
        clone = root.clone()
        path = base + (3, 4) * (delay // 2) + trigger
        for action in path:
            clone.step(action)
        at_trigger = hud_state(clone.frame())
        for action in finish:
            if clone.levels_completed > 6:
                break
            clone.step(action)
        results.append(
            (
                delay,
                at_trigger,
                int(clone.levels_completed),
                avatar_position(clone.frame()),
                meter(clone.frame()),
                hud_state(clone.frame()),
                compact_path(path + finish),
            )
        )
    print("phase_candidates", results)


def exact_candidate(root):
    prepare = state_candidate_path(3, 5)
    transform = (3, 1, 1, 4, 1, 1, 1, 1, 2)
    finish = (3,) + (2,) * 3 + (3,) * 2 + (1,) * 2 + (2,) * 4
    path = prepare + transform + finish
    clone = root.clone()
    checkpoints = []
    for step, action in enumerate(path, 1):
        if clone.levels_completed > 6:
            break
        clone.step(action)
        if step in (len(prepare), len(prepare) + len(transform), len(path)):
            checkpoints.append(
                (
                    step,
                    avatar_position(clone.frame()),
                    meter(clone.frame()),
                    hud_state(clone.frame()),
                    int(clone.levels_completed),
                )
            )
    print(
        "exact_candidate",
        len(path),
        compact_path(path),
        checkpoints,
        int(clone.levels_completed),
    )


def trace(root, name, actions):
    clone = root.clone()
    states = []
    for action in actions:
        clone.step(action)
        states.append(
            (
                int(action),
                avatar_position(clone.frame()),
                meter(clone.frame()),
                world_objects(clone.frame()),
                int(clone.levels_completed),
            )
        )
    print("trace", name, states)


def contextual_actions(root, name, prefix):
    base = root.clone()
    for action in prefix:
        base.step(action)
    before = avatar_position(base.frame())
    outcomes = {}
    for action in base.actions:
        child = base.clone()
        child.step(int(action))
        outcomes[int(action)] = (
            before,
            avatar_position(child.frame()),
            meter(child.frame()),
            int(child.levels_completed),
        )
    print("context", name, compact_path(prefix), outcomes)


def compact_path(path):
    names = {1: "U", 2: "D", 3: "L", 4: "R"}
    runs = []
    for action in path:
        if runs and runs[-1][0] == action:
            runs[-1] = (action, runs[-1][1] + 1)
        else:
            runs.append((action, 1))
    return "".join(f"{names[action]}{count}" for action, count in runs)


def explore(root, max_states=1200, max_depth=90, key_with_meter=True):
    start = avatar_position(root.frame())
    queue = deque([()])
    seen = {(start, meter(root.frame())) if key_with_meter else start}
    positions = {start: ()}
    refills = {}
    jumps = {}
    edges = {}
    hud_edges = {}
    objects = set(world_objects(root.frame()))
    feature_paths = {
        feature: () for feature in world_features(root.frame())
    }
    pattern_paths = {
        pattern: () for pattern in tile_patterns(root.frame())
    }
    win = None

    def reconstruct(path):
        node = root.clone()
        for action in path:
            node.step(action)
        return node

    while queue and len(seen) < max_states:
        path = queue.popleft()
        node = reconstruct(path)
        if len(path) >= max_depth:
            continue
        old_position = avatar_position(node.frame())
        old_meter = meter(node.frame())
        old_hud = hud_state(node.frame())
        for action in node.actions:
            child = node.clone()
            child.step(int(action))
            child_path = path + (int(action),)
            if child.levels_completed > 6:
                win = child_path
                queue.clear()
                break
            new_position = avatar_position(child.frame())
            new_meter = meter(child.frame())
            new_hud = hud_state(child.frame())
            objects.update(world_objects(child.frame()))
            for feature in world_features(child.frame()):
                feature_paths.setdefault(feature, child_path)
            for pattern in tile_patterns(child.frame()):
                pattern_paths.setdefault(pattern, child_path)
            if new_position is None or new_position == old_position:
                continue
            edges.setdefault(old_position, {})[int(action)] = new_position
            if new_hud != old_hud:
                hud_edges.setdefault(
                    (old_position, int(action), new_position, old_hud, new_hud),
                    child_path,
                )
            positions.setdefault(new_position, child_path)
            if new_meter > old_meter:
                refills.setdefault(new_position, child_path)
            distance = (
                abs(new_position[0] - old_position[0])
                + abs(new_position[1] - old_position[1])
            )
            if distance != 5:
                jumps.setdefault((old_position, new_position), child_path)
            key = (new_position, new_meter) if key_with_meter else new_position
            if key in seen:
                continue
            seen.add(key)
            queue.append(child_path)
        if win is not None:
            break
    relative_positions = sorted(
        ((row - start[0]) // 5, (col - start[1]) // 5)
        for row, col in positions
    )
    print(
        "explore",
        "states",
        len(seen),
        "positions",
        len(positions),
        "frontier",
        len(queue),
        "relative",
        relative_positions,
    )
    print(
        "progress",
        "refills",
        {
            ((row - start[0]) // 5, (col - start[1]) // 5): compact_path(path)
            for (row, col), path in refills.items()
        },
        "jumps",
        {
            (
                ((old[0] - start[0]) // 5, (old[1] - start[1]) // 5),
                ((new[0] - start[0]) // 5, (new[1] - start[1]) // 5),
            ): compact_path(path)
            for (old, new), path in jumps.items()
        },
        "objects",
        sorted(objects),
    )
    print(
        "features",
        {
            feature: compact_path(path)
            for feature, path in sorted(feature_paths.items())
        },
    )
    print(
        "tile_patterns",
        {
            pattern: compact_path(path)
            for pattern, path in sorted(pattern_paths.items())
        },
    )
    boundary_paths = {}
    for position, path in positions.items():
        row, col = position
        outward = []
        if row <= 7:
            outward.append("U")
        if row >= 47:
            outward.append("D")
        if col <= 9:
            outward.append("L")
        if col >= 54:
            outward.append("R")
        if outward:
            relative = ((row - start[0]) // 5, (col - start[1]) // 5)
            boundary_paths[(relative, "".join(outward))] = compact_path(path)
    print("boundary_paths", boundary_paths)
    print("reward_path", None if win is None else compact_path(win))
    print(
        "hud_edges",
        [
            (
                ((old[0] - start[0]) // 5, (old[1] - start[1]) // 5),
                action,
                ((new[0] - start[0]) // 5, (new[1] - start[1]) // 5),
                before,
                after,
                compact_path(path),
            )
            for (old, action, new, before, after), path in hud_edges.items()
        ],
    )
    if not key_with_meter and len(refills) >= 6:
        ring_route(root, start, edges, set(refills))


def ring_route(root, start, edges, rings):
    def graph_path(source, target):
        queue = deque([(source, ())])
        seen = {source}
        while queue:
            position, path = queue.popleft()
            if position == target:
                return path
            for action, child in edges.get(position, {}).items():
                if child not in seen:
                    seen.add(child)
                    queue.append((child, path + (action,)))
        return None

    points = [start] + sorted(rings)
    pair_paths = {
        (source, target): graph_path(source, target)
        for source in points
        for target in rings
        if source != target
    }
    candidates = []
    for order in permutations(sorted(rings)):
        path = ()
        position = start
        possible = True
        for target in order:
            segment = pair_paths.get((position, target))
            if segment is None:
                possible = False
                break
            path += segment
            position = target
        if not possible or len(path) > 90:
            continue
        energy = 21
        collected = set()
        position = start
        valid = True
        for action in path:
            position = edges[position][action]
            energy -= 1
            if position in rings and position not in collected:
                collected.add(position)
                energy = 21
            if energy < 0:
                valid = False
                break
            if collected == rings:
                break
        if valid and collected == rings:
            candidates.append((len(path), path, order))
    candidates.sort(key=lambda item: item[0])
    verified = []
    for length, path, order in candidates[:12]:
        clone = root.clone()
        events = []
        for step, action in enumerate(path, 1):
            if clone.levels_completed > 6:
                break
            old_meter = meter(clone.frame())
            old_hud = hud_state(clone.frame())
            clone.step(action)
            new_meter = meter(clone.frame())
            new_hud = hud_state(clone.frame())
            if new_meter > old_meter or new_hud != old_hud:
                events.append(
                    (
                        step,
                        avatar_position(clone.frame()),
                        old_meter,
                        new_meter,
                        None if new_hud == old_hud else new_hud,
                    )
                )
        verified.append(
            (
                length,
                compact_path(path),
                tuple(
                    ((row - start[0]) // 5, (col - start[1]) // 5)
                    for row, col in order
                ),
                int(clone.levels_completed),
                avatar_position(clone.frame()),
                meter(clone.frame()),
                events,
            )
        )
        if clone.levels_completed > 6:
            break
    print("route_plan", verified)


def probe(env):
    enter_level_7(env)
    clone = env.clone()
    if len(sys.argv) > 1 and sys.argv[1] in ("explore", "explore_pos"):
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
        explore(clone, max_states=limit, key_with_meter=sys.argv[1] == "explore")
        return
    if len(sys.argv) > 1 and sys.argv[1] == "lower":
        inspect_lower(clone)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "agents":
        inspect_agents(clone)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "agents_move":
        inspect_agents(clone, moving=True)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "multi":
        inspect_multiagent(clone)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "candidates":
        candidate(clone, "colored_then_black", (3, 3, 2, 2, 2, 2, 2, 4, 4))
        candidate(clone, "colored_black_refill", (3, 3, 2, 2, 2, 2, 2, 4, 4, 3, 2))
        candidate(clone, "colored_from_right", (3, 3, 2, 2, 2, 2, 4, 2, 3))
        candidate(clone, "black_from_above", (3, 3, 2, 2, 2, 2, 4, 4, 2))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "cycles":
        cycle_probe(clone)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "target":
        inspect_target(clone)
        candidate(
            clone,
            "bottom_then_right",
            (1, 1, 4, 4, 2, 4, 2, 2, 4, 4, 4, 2, 2, 4),
        )
        candidate(
            clone,
            "right_then_down",
            (1, 1, 4, 4, 2, 4, 2, 2, 4, 4, 4, 4, 2, 2),
        )
        target_path = (1, 1, 4, 4, 2, 4, 2, 2, 4, 4, 4, 2, 2, 4)
        contextual_actions(clone, "on_target_after_left_entry", target_path)
        candidate(clone, "target_reentry", target_path + (3, 4))
        return
    if len(sys.argv) > 1 and sys.argv[1] == "right_edge":
        inspect_right_edge(clone)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "brute_states":
        brute_states(clone)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "brute_remote":
        brute_remote_states(clone)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "brute_gates":
        brute_target_gates(clone)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "phase":
        phase_candidate(clone)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "exact":
        exact_candidate(clone)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "explore_state":
        states = {
            "color": (3, 3, 2, 2, 2, 2, 2, 4, 2),
            "shape": (3, 3, 2, 2, 2, 2, 4, 4, 2, 3, 2),
            "both": (3, 3, 2, 2, 2, 2, 2, 4, 4, 3, 2),
        }
        prefix = states[sys.argv[2]]
        for action in prefix:
            clone.step(action)
        print("state_root", sys.argv[2], compact_path(prefix), meter(clone.frame()))
        explore(clone, max_states=200, key_with_meter=False)
        return
    print(
        "entry",
        "level",
        int(clone.levels_completed) + 1,
        "actions",
        tuple(int(action) for action in clone.actions),
        "terminal",
        bool(clone.terminal()),
    )
    print("colors", color_counts(clone.frame()))
    print("tiles", "/".join(tile_map(clone.frame())))
    blobs = [
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(clone.frame(), min_area=3)
        if blob.color != 5
    ]
    print("blobs", blobs)
    before = np.asarray(clone.frame())
    for action in clone.actions:
        child = clone.clone()
        child.step(int(action))
        after = np.asarray(child.frame())
        delta = frame_delta(before, after)
        transitions = {}
        for old, new in zip(before[before != after], after[before != after]):
            pair = (int(old), int(new))
            transitions[pair] = transitions.get(pair, 0) + 1
        focus = [
            (blob.color, blob.bbox, blob.area)
            for blob in connected_components(
                child.frame(), colors=(1, 8, 9, 11, 12), min_area=1
            )
        ]
        print(
            "action",
            int(action),
            "changed",
            delta["count"],
            "bbox",
            delta["bbox"],
            "transitions",
            sorted(transitions.items()),
            "tiles",
            "/".join(tile_map(child.frame())),
            "focus",
            focus,
        )
    trace(clone, "vertical_context", (1, 2))
    trace(clone, "horizontal_context", (3, 4))
    trace(clone, "upper_refill_A", (1, 1, 3, 3))
    trace(clone, "upper_refill_B", (3, 3, 1, 1))
    contextual_actions(clone, "on_upper_refill", (1, 1, 3, 3))
    contextual_actions(clone, "start_after_refill", (1, 1, 3, 3, 4, 4, 2, 2))
    trace(clone, "right_portal", (1, 1, 4, 4, 2, 4, 2, 4))
    contextual_actions(clone, "after_right_portal", (1, 1, 4, 4, 2, 4, 2, 4))


arena.run_program("ls20", probe)
