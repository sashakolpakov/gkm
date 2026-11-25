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


def align_reflection_axis(env, divider_colour=10, marker_colour=11, max_moves=32,
                          axis=0):
    """Centre the selected divider on mirrored markers (axis 0 rows, 1 columns).

    Long colour runs isolate the divider when perpendicular axes cross.
    """
    frame = arr(env.frame())[:63, :63]
    marker_rows = (frame == marker_colour).nonzero()[axis]
    if not len(marker_rows):
        return
    target = float(marker_rows.mean())
    initial_level = env.levels_completed

    def distance(e):
        mask = arr(e.frame())[:63, :63] == divider_colour
        rows = (mask.sum(axis=1-axis) > mask.shape[1-axis] / 2).nonzero()[0]
        if not len(rows):
            rows = mask.nonzero()[axis]
        return abs(float(rows.mean()) - target) if len(rows) else float('inf')

    for _ in range(max_moves):
        if env.terminal() or env.levels_completed > initial_level:
            return
        best, score = None, distance(env)
        for action in ((1, 2) if axis == 0 else (3, 4)):
            clone = env.clone()
            safe_step(clone, action)
            candidate = distance(clone)
            if candidate < score:
                best, score = action, candidate
        if best is None:
            return
        safe_step(env, best)


def guide_selected_shape_to_marker(env, source_colour=5, marker_colour=11,
                                   cell=3, max_moves=64, reflection_colour=4):
    """Identify the selected source by motion and translate it onto its outline.

    Sample tile corners to ignore the small control dots inside source tiles.
    Match the source's orientation against marker outlines before steering.
    Combined outlines may contain several templates, with already placed
    reflections covering some marker cells.
    """
    def shapes(frame, colour):
        f = arr(frame)[:63, :63]
        result = []
        for blob in connected_components(f, colors=[colour]):
            r0, c0, r1, c1 = blob.bbox
            outline = tuple((r - r0, c - c0)
                            for r in range(r0, r1 + 1, cell)
                            for c in range(c0, c1 + 1, cell)
                            if f[r, c] == colour)
            result.append((blob, outline))
        return result

    initial_level = env.levels_completed
    frame = arr(env.frame())[:63, :63]
    sources = shapes(frame, source_colour)
    selected = None
    for action in (1, 2, 3, 4):
        clone = env.clone()
        safe_step(clone, action)
        after = arr(clone.frame())[:63, :63]
        changes = []
        for blob, outline in sources:
            r0, c0, r1, c1 = blob.bbox
            region = (slice(r0, r1 + 1), slice(c0, c1 + 1))
            changes.append(int(((frame[region] == source_colour) &
                                (after[region] != source_colour)).sum()))
        if changes and max(changes) > 0:
            selected = sources[changes.index(max(changes))]
            break
    if selected is None:
        return
    blob, outline = selected
    position = blob.top_left
    targets = [b.top_left for b, signature in shapes(frame, marker_colour)
               if signature == outline]
    if not targets:
        height = max(r for r, c in outline) + cell
        width = max(c for r, c in outline) + cell
        targets = [(r, c)
                   for r in range(0, frame.shape[0] - height + 1, cell)
                   for c in range(0, frame.shape[1] - width + 1, cell)
                   if all(frame[r + dr, c + dc] in
                          (marker_colour, reflection_colour)
                          for dr, dc in outline)
                   and any(frame[r + dr, c + dc] == marker_colour
                           for dr, dc in outline)]
    if not targets:
        return
    target = min(targets, key=lambda p: sum(abs(a-b) for a, b in zip(p, position)))

    def locate(e, previous):
        # Touching sources become one colour component; retain the selected
        # outline as a template rather than treating that union as one object.
        f = arr(e.frame())[:63, :63]
        height = max(r for r, c in outline) + cell
        width = max(c for r, c in outline) + cell
        matches = [(r, c)
                   for r in range(0, f.shape[0] - height + 1, cell)
                   for c in range(0, f.shape[1] - width + 1, cell)
                   if all(f[r + dr, c + dc] == source_colour
                          for dr, dc in outline)]
        return min(matches, key=lambda p: sum(abs(a-b) for a, b in zip(p, previous))) if matches else None

    for _ in range(max_moves):
        if env.terminal() or env.levels_completed > initial_level:
            return
        score = sum(abs(a-b) for a, b in zip(position, target))
        best, next_position = None, position
        for action in (1, 2, 3, 4):
            clone = env.clone()
            safe_step(clone, action)
            if clone.levels_completed > initial_level:
                best = action
                break
            candidate = locate(clone, position)
            if candidate is None:
                continue
            distance = sum(abs(a-b) for a, b in zip(candidate, target))
            if distance < score:
                best, next_position, score = action, candidate, distance
        if best is None:
            return
        safe_step(env, best)
        position = next_position
