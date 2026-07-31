# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.

from collections import deque

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
