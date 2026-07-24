# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.

from perception import arr, connected_components


def click_coordinate(env, x, y, times=1, action=6):
    """Click one screen coordinate repeatedly, stopping after level progress."""
    base_level = env.levels_completed
    for _ in range(times):
        env.step(action, int(x), int(y))
        if env.terminal() or env.levels_completed > base_level:
            break


def align_slider_tips_to_hollow_targets(env, marker_color=13, action=6):
    """Align orthogonal colored slider tips with same-axis hollow targets."""

    def tips_and_targets(frame):
        grid = arr(frame)
        rows, cols = grid.shape[:2]
        tips = []
        targets = []
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                neighbors = (
                    int(grid[r - 1, c]),
                    int(grid[r + 1, c]),
                    int(grid[r, c - 1]),
                    int(grid[r, c + 1]),
                )
                if int(grid[r, c]) == marker_color:
                    if len(set(neighbors)) == 1 and neighbors[0] != marker_color:
                        tips.append((r, c, neighbors[0]))
                elif all(color == marker_color for color in neighbors):
                    targets.append((r, c))
        return tips, targets

    frame = env.frame()
    tips, targets = tips_and_targets(frame)
    plans = []
    for tip_r, tip_c, body_color in tips:
        aligned_targets = [
            target
            for target in targets
            if target[0] == tip_r or target[1] == tip_c
        ]
        if not aligned_targets:
            continue
        target_r, target_c = min(
            aligned_targets,
            key=lambda target: abs(target[0] - tip_r) + abs(target[1] - tip_c),
        )
        initial_distance = abs(target_r - tip_r) + abs(target_c - tip_c)

        buttons = []
        for blob in connected_components(frame, colors=(body_color,)):
            r0, c0, r1, c1 = blob.bbox
            if r0 <= tip_r <= r1 and c0 <= tip_c <= c1:
                continue
            buttons.append((int(round(blob.centroid[1])), int(round(blob.centroid[0]))))

        best = None
        for x, y in buttons:
            clone = env.clone()
            clone.step(action, x, y)
            moved_tips, _ = tips_and_targets(clone.frame())
            matching = [
                (r, c)
                for r, c, color in moved_tips
                if color == body_color
            ]
            if not matching:
                continue
            moved_r, moved_c = min(
                matching,
                key=lambda point: abs(point[0] - tip_r) + abs(point[1] - tip_c),
            )
            distance = abs(target_r - moved_r) + abs(target_c - moved_c)
            candidate = (distance, x, y)
            if best is None or candidate < best:
                best = candidate

        if best is None or best[0] >= initial_distance:
            continue
        improvement = initial_distance - best[0]
        if initial_distance % improvement == 0:
            plans.append((best[1], best[2], initial_distance // improvement))

    for x, y, times in plans:
        click_coordinate(env, x, y, times, action)
