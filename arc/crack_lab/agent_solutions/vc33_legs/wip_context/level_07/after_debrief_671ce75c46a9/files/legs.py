# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.

from collections import deque
import heapq

from perception import arr, bounded_replay_bfs, connected_components


def align_lower_cyan_tip_with_upper_notch(
    env, max_presses=12, marker_color=11, max_states=300
):
    """Use visible controls to horizontally align a lower marker to an upper notch."""
    start_level = env.levels_completed

    def marker_distance(node):
        markers = connected_components(
            node.frame(), colors=(marker_color,), min_area=4
        )
        if len(markers) < 2:
            return None
        markers.sort(key=lambda blob: blob.centroid[0])
        upper, lower = markers[0], markers[-1]
        return abs(lower.centroid[1] - upper.centroid[1])

    def control_actions(node):
        actions = []
        for control in connected_components(
            node.frame(), colors=(9,), min_area=4
        ):
            r0, c0, r1, c1 = control.bbox
            actions.append((6, (c0 + c1) // 2, (r0 + r1) // 2))
        return actions

    def aligned(node, _path):
        if node.levels_completed > start_level:
            return True
        distance = marker_distance(node)
        return distance is not None and distance == 0

    path = bounded_replay_bfs(
        env,
        aligned,
        control_actions,
        key_fn=lambda node: (
            node.levels_completed,
            arr(node.frame())[1:].tobytes(),
        ),
        max_states=max_states,
        max_depth=max_presses,
    )
    if path is not None:
        for action in path:
            if env.levels_completed > start_level or env.terminal():
                return
            env.step(*action)
        return

    # Retain a cheap greedy fallback for unusually large control configurations.
    for _ in range(max_presses):
        if env.levels_completed > start_level or env.terminal():
            return
        current = marker_distance(env)
        controls = connected_components(env.frame(), colors=(9,), min_area=4)
        if current is None or not controls:
            return

        best = None
        for control in controls:
            r0, c0, r1, c1 = control.bbox
            x, y = (c0 + c1) // 2, (r0 + r1) // 2
            clone = env.clone()
            clone.step(6, x, y)
            distance = marker_distance(clone)
            if clone.levels_completed > start_level:
                best = (-1, x, y)
                break
            if distance is not None and (best is None or distance < best[0]):
                best = (distance, x, y)

        if best is None or best[0] >= current:
            return
        env.step(6, best[1], best[2])


def relay_height_between_adjacent_reservoirs(env, max_presses=32):
    """Relay height leftward until every small same-color marker pair is level."""
    start_level = env.levels_completed
    frame = env.frame()
    excluded = {0, 3, 4, 5, 7, 9}
    by_color = {}
    for blob in connected_components(frame, min_area=1):
        if blob.color not in excluded and blob.area <= 12:
            by_color.setdefault(blob.color, []).append(blob)
    pairs = {
        color: blobs for color, blobs in by_color.items() if len(blobs) == 2
    }
    if not pairs:
        return

    controls = connected_components(frame, colors=(9,), min_area=4)
    controls.sort(key=lambda blob: blob.centroid[1])
    if len(controls) < 2:
        return
    actions = [
        (6, int(blob.centroid[1]), int(blob.centroid[0]))
        for blob in controls[::2]
    ]

    def gaps(node):
        out = {}
        for color in pairs:
            blobs = connected_components(
                node.frame(), colors=(color,), min_area=1
            )
            if len(blobs) == 2:
                out[color] = abs(blobs[0].centroid[0] - blobs[1].centroid[0])
        return out

    left_color = min(
        pairs,
        key=lambda color: min(blob.centroid[1] for blob in pairs[color]),
    )
    presses = 0

    # Empty each intervening reservoir only as needed to feed the left target.
    while presses < max_presses:
        current_gaps = gaps(env)
        if current_gaps.get(left_color) == 0:
            break
        before = arr(env.frame())[1:].tobytes()
        moved = False
        for action in actions[:-1]:
            clone = env.clone()
            clone.step(*action)
            if arr(clone.frame())[1:].tobytes() == before:
                continue
            env.step(*action)
            presses += 1
            moved = True
            break
        if not moved:
            return

    # The final reservoir supplies the remaining marked reservoir(s). Only
    # commit transfers that strictly reduce visible marker mismatch.
    if not actions:
        return
    while presses < max_presses:
        if env.levels_completed > start_level or env.terminal():
            return
        current = gaps(env)
        if current and all(gap == 0 for gap in current.values()):
            return
        action = actions[-1]
        clone = env.clone()
        clone.step(*action)
        if clone.levels_completed > start_level:
            env.step(*action)
            return
        after = gaps(clone)
        if not after or sum(after.values()) >= sum(current.values()):
            return
        env.step(*action)
        presses += 1


def cross_pressure_gates_then_align_height(
    env,
    marker_color=11,
    closed_gate_color=1,
    active_gate_colors=(12, 13, 14, 15),
    max_stages=10,
    max_states=700,
    max_depth=16,
):
    """Move a marked platform through pressure gates, then align its height."""
    start_level = env.levels_completed
    gate_colors = (closed_gate_color,) + tuple(active_gate_colors)

    def marker_pair(node):
        markers = connected_components(
            node.frame(), colors=(marker_color,), min_area=2
        )
        if len(markers) != 2 or markers[0].area == markers[1].area:
            return None
        moving = max(markers, key=lambda blob: blob.area)
        fixed = min(markers, key=lambda blob: blob.area)
        return moving, fixed

    def progress(node):
        pair = marker_pair(node)
        if pair is None:
            return None
        moving, fixed = pair
        gates = [
            blob
            for blob in connected_components(
                node.frame(), colors=gate_colors, min_area=2
            )
            if moving.centroid[1] < blob.centroid[1] < fixed.centroid[1]
        ]
        return (
            moving.centroid[1],
            len(gates),
            abs(moving.centroid[0] - fixed.centroid[0]),
        )

    def actions(node):
        options = [
            (6, int(blob.centroid[1]), int(blob.centroid[0]))
            for blob in connected_components(
                node.frame(), colors=(9,), min_area=2
            )
        ]
        options.extend(
            (6, int(blob.centroid[1]), int(blob.centroid[0]))
            for blob in connected_components(
                node.frame(), colors=active_gate_colors, min_area=2
            )
        )
        return options

    for _ in range(max_stages):
        if env.levels_completed > start_level or env.terminal():
            return
        initial = progress(env)
        if initial is None:
            return
        start_column, gates_left, start_gap = initial

        def improved(node):
            if node.levels_completed > start_level:
                return True
            current = progress(node)
            if current is None:
                return False
            column, remaining, gap = current
            if gates_left:
                return column > start_column
            return column == start_column and remaining == 0 and gap < start_gap

        queue = deque([(env.clone(), [])])
        seen = {
            (
                env.levels_completed,
                arr(env.frame())[1:].tobytes(),
            )
        }
        stage_path = None
        while queue and len(seen) <= max_states:
            node, path = queue.popleft()
            if len(path) >= max_depth or node.terminal():
                continue
            for action in actions(node):
                child = node.clone()
                child.step(*action)
                child_path = path + [action]
                if improved(child):
                    stage_path = child_path
                    queue.clear()
                    break
                key = (
                    child.levels_completed,
                    arr(child.frame())[1:].tobytes(),
                )
                if key not in seen:
                    seen.add(key)
                    queue.append((child, child_path))
        if stage_path is None:
            return
        for action in stage_path:
            if env.levels_completed > start_level or env.terminal():
                return
            env.step(*action)


def cross_horizontal_gates_then_align_opposing_markers(
    env,
    marker_colors=(11, 14),
    closed_gate_color=1,
    active_gate_colors=(12, 13, 14, 15),
    max_stages=14,
    max_states=2500,
    max_depth=24,
):
    """Relay opposing marked platforms across horizontal pressure gates."""
    start_level = env.levels_completed
    gate_colors = (closed_gate_color,) + tuple(active_gate_colors)

    def marker_pair(node, color):
        markers = connected_components(
            node.frame(), colors=(color,), min_area=2
        )
        if len(markers) != 2 or markers[0].area == markers[1].area:
            return None
        return (
            max(markers, key=lambda blob: blob.area),
            min(markers, key=lambda blob: blob.area),
        )

    def progress(node, color):
        pair = marker_pair(node, color)
        if pair is None:
            return None
        moving, fixed = pair
        low, high = sorted((moving.centroid[0], fixed.centroid[0]))
        gates = [
            blob
            for blob in connected_components(
                node.frame(), colors=gate_colors, min_area=12
            )
            if (
                blob.size[1] > blob.size[0]
                and low < blob.centroid[0] < high
            )
        ]
        return (
            len(gates),
            abs(moving.centroid[0] - fixed.centroid[0]),
            abs(moving.centroid[1] - fixed.centroid[1]),
        )

    def actions(node):
        options = [
            (6, int(blob.centroid[1]), int(blob.centroid[0]))
            for blob in connected_components(
                node.frame(), colors=(9,), min_area=2
            )
        ]
        options.extend(
            (6, int(blob.centroid[1]), int(blob.centroid[0]))
            for blob in connected_components(
                node.frame(), colors=active_gate_colors, min_area=12
            )
            if blob.size[1] > blob.size[0]
        )
        return options

    # Platforms moving upward are staged first; their displaced pressure
    # becomes the supply that lets the opposing platforms travel downward.
    colors = []
    for color in marker_colors:
        pair = marker_pair(env, color)
        if pair is not None:
            moving, fixed = pair
            colors.append((moving.centroid[0] < fixed.centroid[0], color))
    ordered_colors = [color for _, color in sorted(colors)]

    for color_index, color in enumerate(ordered_colors):
        for _ in range(max_stages):
            if env.levels_completed > start_level or env.terminal():
                return
            initial = progress(env, color)
            if initial is None:
                break
            if initial[0] == 0 and initial[2] == 0 and color_index + 1 < len(
                ordered_colors
            ):
                break

            def improved(node):
                if node.levels_completed > start_level:
                    return True
                current = progress(node, color)
                if current is None:
                    return False
                if initial[0]:
                    return current[0] < initial[0]
                return current[0] == 0 and current[2] < initial[2]

            queue = deque([(env.clone(), [])])
            seen = {
                (
                    env.levels_completed,
                    arr(env.frame())[1:].tobytes(),
                )
            }
            stage_path = None
            while queue and len(seen) <= max_states:
                node, path = queue.popleft()
                if len(path) >= max_depth or node.terminal():
                    continue
                for action in actions(node):
                    child = node.clone()
                    child.step(*action)
                    child_path = path + [action]
                    if improved(child):
                        stage_path = child_path
                        queue.clear()
                        break
                    key = (
                        child.levels_completed,
                        arr(child.frame())[1:].tobytes(),
                    )
                    if key not in seen:
                        seen.add(key)
                        queue.append((child, child_path))
            if stage_path is None:
                break
            for action in stage_path:
                if env.levels_completed > start_level or env.terminal():
                    return
                env.step(*action)


def align_marker_pair_with_pressure_controls(
    env,
    marker_color=11,
    active_gate_colors=(12, 13, 14, 15),
    max_stages=16,
    max_states=500,
    max_depth=12,
):
    """Align an unequal marker pair using visible pressure controls and gates."""
    start_level = env.levels_completed

    def marker_gap(node):
        markers = connected_components(
            node.frame(), colors=(marker_color,), min_area=2
        )
        if len(markers) != 2 or markers[0].area == markers[1].area:
            return None
        moving = max(markers, key=lambda blob: blob.area)
        fixed = min(markers, key=lambda blob: blob.area)
        return (
            abs(moving.centroid[0] - fixed.centroid[0])
            + abs(moving.centroid[1] - fixed.centroid[1])
        )

    def actions(node):
        return [
            (6, int(blob.centroid[1]), int(blob.centroid[0]))
            for blob in connected_components(
                node.frame(),
                colors=(9,) + tuple(active_gate_colors),
                min_area=2,
            )
        ]

    for _ in range(max_stages):
        if env.levels_completed > start_level or env.terminal():
            return
        initial_gap = marker_gap(env)
        if initial_gap is None:
            return

        queue = deque([(env.clone(), [])])
        seen = {arr(env.frame())[1:].tobytes()}
        stage_path = None
        while queue and len(seen) <= max_states:
            node, path = queue.popleft()
            if len(path) >= max_depth or node.terminal():
                continue
            for action in actions(node):
                child = node.clone()
                child_path = path + [action]
                child.step(*action)
                gap = marker_gap(child)
                if (
                    child.levels_completed > start_level
                    or (gap is not None and gap < initial_gap)
                ):
                    stage_path = child_path
                    queue.clear()
                    break
                key = arr(child.frame())[1:].tobytes()
                if key not in seen:
                    seen.add(key)
                    queue.append((child, child_path))
        if stage_path is None:
            return
        for action in stage_path:
            if env.levels_completed > start_level or env.terminal():
                return
            env.step(*action)


def cross_pressure_gate_toward_matching_marker(
    env,
    marker_color=11,
    closed_gate_color=1,
    active_gate_colors=(12, 13, 14, 15),
    max_states=1800,
    max_depth=18,
):
    """Cross the nearest pressure gate in the direction of a matching marker."""
    start_level = env.levels_completed
    gate_colors = (closed_gate_color,) + tuple(active_gate_colors)

    def marker_pair(node):
        markers = [
            blob
            for blob in connected_components(
                node.frame(), colors=(marker_color,), min_area=2
            )
            if blob.area <= 5
        ]
        if len(markers) != 2 or markers[0].area == markers[1].area:
            return None
        return (
            max(markers, key=lambda blob: blob.area),
            min(markers, key=lambda blob: blob.area),
        )

    initial_pair = marker_pair(env)
    if initial_pair is None:
        return
    initial_moving, fixed = initial_pair
    start_column = initial_moving.centroid[1]
    direction = 1 if fixed.centroid[1] > start_column else -1

    gates = [
        blob
        for blob in connected_components(
            env.frame(), colors=gate_colors, min_area=8
        )
        if (
            blob.size[0] > blob.size[1]
            and direction * (blob.centroid[1] - start_column) > 0
        )
    ]
    if not gates:
        return
    target_gate = min(
        gates, key=lambda blob: abs(blob.centroid[1] - start_column)
    )
    target_bbox = target_gate.bbox
    target_row = target_bbox[0]

    pad_rows = sorted(
        {
            int(blob.centroid[0])
            for blob in connected_components(
                env.frame(), colors=(9,), min_area=2
            )
        }
    )
    lower_tier = pad_rows[-1] if len(pad_rows) > 1 else 32

    gate_columns = [
        blob.centroid[1]
        for blob in connected_components(
            env.frame(), colors=gate_colors, min_area=8
        )
        if blob.size[0] > blob.size[1]
    ]
    left_gate_column = min(gate_columns)
    right_gate_column = max(gate_columns)
    center_column = int((left_gate_column + right_gate_column) / 2)

    def surface_at(node, column, row_start, row_stop):
        frame = arr(node.frame())
        for row in range(row_start, row_stop):
            if int(frame[row, column]) in (3, 11, 14, 15):
                return row
        return row_stop

    def pressure_gap(node):
        gate_column = (target_bbox[1] + target_bbox[3]) / 2
        center_surface = surface_at(node, center_column, 1, 63)
        if target_row >= lower_tier:
            outer_start, outer_stop = lower_tier, 63
        else:
            outer_start = pad_rows[0] if pad_rows else 1
            outer_stop = lower_tier
        if abs(gate_column - right_gate_column) <= abs(
            gate_column - left_gate_column
        ):
            outer_column = target_bbox[3] + 1
        else:
            outer_column = target_bbox[1] - 1
        outer_surface = surface_at(
            node, outer_column, outer_start, outer_stop
        )
        return abs(center_surface - target_row) + abs(
            outer_surface - target_row
        )

    def crossed(node):
        if node.levels_completed > start_level:
            return True
        pair = marker_pair(node)
        if pair is None:
            return False
        moving, _ = pair
        return direction * (moving.centroid[1] - start_column) > 4

    def actions(node):
        options = [
            (6, int(blob.centroid[1]), int(blob.centroid[0]))
            for blob in connected_components(
                node.frame(), colors=(9,), min_area=2
            )
        ]
        options.extend(
            (6, int(blob.centroid[1]), int(blob.centroid[0]))
            for blob in connected_components(
                node.frame(), colors=active_gate_colors, min_area=8
            )
            if blob.bbox == target_bbox
        )
        return options

    root = env.clone()
    counter = 0
    queue = [(pressure_gap(root), 0, counter, root, [])]
    seen = {arr(root.frame())[1:].tobytes()}
    solution = None
    while queue and len(seen) <= max_states:
        _, depth, _, node, path = heapq.heappop(queue)
        if depth >= max_depth or node.terminal():
            continue
        for action in actions(node):
            child = node.clone()
            child.step(*action)
            child_path = path + [action]
            if crossed(child):
                solution = child_path
                queue.clear()
                break
            key = arr(child.frame())[1:].tobytes()
            if key in seen:
                continue
            seen.add(key)
            counter += 1
            heapq.heappush(
                queue,
                (
                    pressure_gap(child),
                    depth + 1,
                    counter,
                    child,
                    child_path,
                ),
            )
    if solution is None:
        return
    for action in solution:
        if env.levels_completed > start_level or env.terminal():
            return
        env.step(*action)


def cross_pressure_gates_toward_matching_markers(
    env,
    marker_colors,
    closed_gate_color=1,
    active_gate_colors=(12, 13, 14, 15),
    max_states=1800,
    max_depth=18,
):
    """Cross one pressure gate for each matching-marker color, in order."""
    for marker_color in marker_colors:
        cross_pressure_gate_toward_matching_marker(
            env,
            marker_color=marker_color,
            closed_gate_color=closed_gate_color,
            active_gate_colors=active_gate_colors,
            max_states=max_states,
            max_depth=max_depth,
        )


def balance_marked_reservoirs_through_center(
    env, marker_colors=(11, 14, 15), max_presses=24
):
    """Balance marked outer reservoirs through one marked central reservoir."""
    start_level = env.levels_completed

    def marker_pairs(node):
        result = {}
        for color in marker_colors:
            markers = [
                blob
                for blob in connected_components(
                    node.frame(), colors=(color,), min_area=2
                )
                if blob.area <= 5
            ]
            if len(markers) == 2 and markers[0].area != markers[1].area:
                result[color] = (
                    max(markers, key=lambda blob: blob.area),
                    min(markers, key=lambda blob: blob.area),
                )
        return result

    controls = [
        (6, int(blob.centroid[1]), int(blob.centroid[0]))
        for blob in connected_components(
            env.frame(), colors=(9,), min_area=2
        )
    ]
    gates = [
        blob
        for blob in connected_components(
            env.frame(), colors=(1, 12, 13, 14, 15), min_area=8
        )
        if blob.size[0] > blob.size[1]
    ]
    if len(gates) < 2:
        return
    left_column = min(blob.centroid[1] for blob in gates)
    right_column = max(blob.centroid[1] for blob in gates)

    pairs = marker_pairs(env)
    central = [
        (color, moving, fixed)
        for color, (moving, fixed) in pairs.items()
        if left_column < moving.centroid[1] < right_column
    ]
    outer = [
        (color, moving, fixed)
        for color, (moving, fixed) in pairs.items()
        if not (left_column < moving.centroid[1] < right_column)
    ]
    if len(central) != 1 or not outer:
        return

    center_color, center_moving, center_fixed = central[0]
    center_row = round(center_moving.centroid[0])
    center_target = round(center_fixed.centroid[0])
    projected_center = center_row - sum(
        round(fixed.centroid[0]) - round(moving.centroid[0])
        for _, moving, fixed in outer
    )
    compensation = center_target - projected_center

    def moving_rows(node):
        return {
            color: round(pair[0].centroid[0])
            for color, pair in marker_pairs(node).items()
        }

    def choose_control(node, wanted_changes):
        before = moving_rows(node)
        for action in controls:
            child = node.clone()
            child.step(*action)
            after = moving_rows(child)
            if all(
                after.get(color, before.get(color)) - before.get(color, 0)
                == change
                for color, change in wanted_changes.items()
            ):
                unchanged = set(before) - set(wanted_changes)
                if all(after.get(color) == before[color] for color in unchanged):
                    return action
        return None

    for _ in range(min(max_presses, abs(compensation) // 2)):
        change = 2 if compensation > 0 else -2
        action = choose_control(env, {center_color: change})
        if action is None:
            return
        env.step(*action)
        if env.levels_completed > start_level or env.terminal():
            return

    for color, _, fixed in outer:
        target = round(fixed.centroid[0])
        for _ in range(max_presses):
            rows = moving_rows(env)
            current = rows.get(color)
            if current is None or current == target:
                break
            change = 2 if target > current else -2
            action = choose_control(
                env, {color: change, center_color: -change}
            )
            if action is None:
                return
            env.step(*action)
            if env.levels_completed > start_level or env.terminal():
                return
