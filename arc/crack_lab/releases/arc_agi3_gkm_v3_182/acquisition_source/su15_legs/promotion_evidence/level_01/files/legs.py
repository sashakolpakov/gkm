"""Shared skills inferred and verified through public frame/action observations."""
from perception import connected_components, safe_step


def click_walk_to_region(env, marker_color=15, goal_color=9,
                         stride=6, max_clicks=32):
    """Walk a field marker into a colored region with short coordinate clicks.

    Plan on a clone, measuring distance to the region's center. Commit only a
    path that increases the level reward. Header objects are excluded from the
    movable marker candidates.
    """
    trial = env.clone()
    base_level = env.levels_completed
    goals = connected_components(trial.frame(), colors=[goal_color], min_area=4)
    markers = [b for b in connected_components(
        trial.frame(), colors=[marker_color], min_area=4) if b.bbox[0] >= 10]
    if not goals or not markers:
        return
    target = max(goals, key=lambda b: b.area).centroid
    marker = max(markers, key=lambda b: b.area).centroid
    path = []
    for _ in range(max_clicks):
        y, x = marker
        ty, tx = target
        nx = round(x + max(-stride, min(stride, tx - x)))
        ny = round(y + max(-stride, min(stride, ty - y)))
        action = (6, nx, ny)
        safe_step(trial, action)
        path.append(action)
        if trial.levels_completed > base_level:
            for action in path:
                safe_step(env, action)
                if env.levels_completed > base_level or env.terminal():
                    return
            return
        markers = [b for b in connected_components(
            trial.frame(), colors=[marker_color], min_area=4)
                   if b.bbox[0] >= 10]
        if not markers:
            return
        next_marker = min(markers, key=lambda b:
                          (b.centroid[0]-y)**2 + (b.centroid[1]-x)**2).centroid
        before = (ty-y)**2 + (tx-x)**2
        after = (ty-next_marker[0])**2 + (tx-next_marker[1])**2
        if after >= before:
            return
        marker = next_marker
