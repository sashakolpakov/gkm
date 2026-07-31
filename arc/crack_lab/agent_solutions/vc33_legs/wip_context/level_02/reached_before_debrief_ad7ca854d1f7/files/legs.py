# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.

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
