"""Shared skills inferred and verified through public frame/action observations."""
from perception import connected_components, safe_step


def click_walk_to_region(env, marker_color=15, goal_color=9,
                         stride=6, max_clicks=32, goal_index=None,
                         arrival_radius=None):
    """Walk a field marker into a colored region with short coordinate clicks.

    Plan on a clone, measuring distance to the region's center. Commit only a
    path that increases the level reward, or reaches an explicitly requested
    arrival radius for a partial placement. Goal indices run left to right.
    Header objects are excluded from the movable marker candidates.
    """
    trial = env.clone()
    base_level = env.levels_completed
    goals = connected_components(trial.frame(), colors=[goal_color], min_area=4)
    markers = [b for b in connected_components(
        trial.frame(), colors=[marker_color], min_area=4) if b.bbox[0] >= 10]
    if not goals or not markers:
        return
    target = (max(goals, key=lambda b: b.area) if goal_index is None else
              sorted(goals, key=lambda b: b.centroid[1])[goal_index]).centroid
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
        if arrival_radius is not None and after <= arrival_radius**2:
            for action in path:
                safe_step(env, action)
                if env.levels_completed > base_level or env.terminal():
                    return
            return
        if after >= before:
            return
        marker = next_marker


def merge_matching_markers(env, colors=(10, 6, 15, 11), stride=3,
                           max_clicks=96, target_count=1):
    """Bring nearest equal field markers together to a requested count.

    Short clicks move nearby markers; equal neighbors fuse into the next size.
    Plan on a clone and commit only after the marker-count subgoal is met.
    """
    from math import hypot

    def markers(world):
        return [b for b in connected_components(world.frame(), colors=colors)
                if b.bbox[0] >= 10]

    trial = env.clone()
    path = []
    for _ in range(max_clicks):
        blobs = markers(trial)
        if len(blobs) == target_count:
            for action in path:
                safe_step(env, action)
            return
        pairs = [(hypot(a.centroid[0] - b.centroid[0],
                        a.centroid[1] - b.centroid[1]), a, b)
                 for i, a in enumerate(blobs) for b in blobs[i + 1:]
                 if a.color == b.color]
        if not pairs:
            return
        _, source, target = min(pairs, key=lambda pair: pair[0])
        # The observed marker anchor is the lower-right pixel of its square.
        y, x = source.bbox[2:]
        ty, tx = target.bbox[2:]
        distance = hypot(ty - y, tx - x)
        step = min(stride, distance)
        action = (6, round(x + (tx - x) * step / (distance or 1)),
                  round(y + (ty - y) * step / (distance or 1)))
        before = [(b.color, b.bbox) for b in blobs]
        safe_step(trial, action)
        path.append(action)
        if [(b.color, b.bbox) for b in markers(trial)] == before:
            return
