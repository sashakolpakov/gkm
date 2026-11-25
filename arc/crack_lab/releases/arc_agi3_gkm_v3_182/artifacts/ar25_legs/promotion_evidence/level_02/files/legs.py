"""Small reusable skills driven only by public frame observations."""
from perception import arr, connected_components, safe_step


def switch_control(env):
    """Switch between the divider and the movable source shape."""
    safe_step(env, 5)


def guide_colour_to_marker(env, moving_colour, marker_colour, max_moves=64):
    """Guide a visible shape to a stationary marker using cloned move deltas.

    The dense objective is Manhattan distance between their bounding-box origins.
    Exclude the final row and column, which contain the observed status strips.
    """
    def origin(frame, colour):
        blobs = connected_components(arr(frame)[:63, :63], colors=[colour])
        if not blobs:
            return None
        return max(blobs, key=lambda blob: blob.area).top_left

    initial_level = env.levels_completed
    target = origin(env.frame(), marker_colour)
    if target is None:
        return

    def distance(frame):
        position = origin(frame, moving_colour)
        if position is None:
            return float('inf')
        return sum(abs(a - b) for a, b in zip(position, target))

    for _ in range(max_moves):
        if env.terminal() or env.levels_completed > initial_level:
            return
        best_action = None
        best_distance = distance(env.frame())
        for action in (1, 2, 3, 4):
            if action not in env.actions:
                continue
            clone = env.clone()
            safe_step(clone, action)
            if clone.levels_completed > initial_level:
                best_action = action
                break
            candidate_distance = distance(clone.frame())
            if candidate_distance < best_distance:
                best_action, best_distance = action, candidate_distance
        if best_action is None:
            return
        safe_step(env, best_action)
