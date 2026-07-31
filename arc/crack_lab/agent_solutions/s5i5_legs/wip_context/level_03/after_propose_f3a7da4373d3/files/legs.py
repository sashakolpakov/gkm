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


def move_articulated_marker_around_barrier(env, marker_color=13, action=6):
    """Use paired limb controls to carry a marker around a blocking wall."""

    def marker_and_target(frame):
        grid = arr(frame)
        rows, cols = grid.shape[:2]
        targets = []
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                if int(grid[r, c]) == marker_color:
                    continue
                if all(
                    int(grid[rr, cc]) == marker_color
                    for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                ):
                    targets.append((r, c))
        if len(targets) != 1:
            return None, None
        target = targets[0]
        target_outline = {
            (target[0] - 1, target[1]),
            (target[0] + 1, target[1]),
            (target[0], target[1] - 1),
            (target[0], target[1] + 1),
        }
        markers = [
            (int(r), int(c))
            for r, c in zip(*((grid == marker_color).nonzero()))
            if (int(r), int(c)) not in target_outline
        ]
        return (markers[0] if len(markers) == 1 else None), target

    def trial(button):
        before, _ = marker_and_target(env.frame())
        clone = env.clone()
        clone.step(action, button[0], button[1])
        if clone.levels_completed > env.levels_completed:
            return True, (0, 0)
        after, _ = marker_and_target(clone.frame())
        if before is None or after is None:
            return False, (0, 0)
        return False, (after[0] - before[0], after[1] - before[1])

    frame = env.frame()
    buttons = sorted(
        (
            int(round(blob.centroid[1])),
            int(round(blob.centroid[0])),
        )
        for blob in connected_components(frame, colors=(4,))
        if blob.bbox[0] >= 50
    )
    directional = {}
    for button in buttons:
        won, delta = trial(button)
        if won:
            click_coordinate(env, *button, action=action)
            return
        if delta != (0, 0):
            directional.setdefault(delta, []).append(button)

    right = sorted(
        button
        for delta, found in directional.items()
        if delta[1] > 0 and delta[0] == 0
        for button in found
    )
    upward = [
        button
        for delta, found in directional.items()
        if delta[0] < 0 and delta[1] == 0
        for button in found
    ]
    downward = [
        button
        for delta, found in directional.items()
        if delta[0] > 0 and delta[1] == 0
        for button in found
    ]
    if len(right) != 2 or len(upward) != 1 or len(downward) != 1:
        return
    near_brace, far_brace = right
    up = upward[0]
    down = downward[0]

    def can_move(button, dr=0, dc=0):
        won, delta = trial(button)
        return won or (delta[0] * dr + delta[1] * dc > 0)

    def advance(button):
        click_coordinate(env, *button, action=action)

    # Brace against the near side of the wall.
    for _ in range(32):
        if not can_move(near_brace, dc=1):
            break
        advance(near_brace)

    # Climb just far enough that the same brace can pass the wall.
    for _ in range(32):
        if can_move(near_brace, dc=1):
            break
        if not can_move(up, dr=-1):
            return
        advance(up)

    # Cross above the wall, switching to the opposing horizontal brace.
    for button in (near_brace, far_brace):
        for _ in range(32):
            marker, target = marker_and_target(env.frame())
            if marker is None or target is None or marker[1] >= target[1]:
                break
            if not can_move(button, dc=1):
                break
            advance(button)

    # Descend along the far side until the marker enters the target.
    for _ in range(32):
        if env.terminal():
            break
        marker, target = marker_and_target(env.frame())
        if marker is None or target is None or marker[0] >= target[0]:
            break
        if not can_move(down, dr=1):
            break
        advance(down)


def dock_crossed_sliders_through_coupled_barriers(
    env,
    horizontal_slider_color=10,
    vertical_slider_color=7,
    upper_limb_color=8,
    upper_joint_color=9,
    lower_limb_color=12,
    lower_joint_color=14,
    action=6,
):
    """Stage two coupled barriers so crossed marker sliders can dock."""
    base_level = env.levels_completed
    control_colors = (
        horizontal_slider_color,
        vertical_slider_color,
        upper_limb_color,
        upper_joint_color,
        lower_limb_color,
        lower_joint_color,
    )
    controls = {}
    for color in control_colors:
        buttons = [
            blob
            for blob in connected_components(
                env.frame(), colors={color}, min_area=5
            )
            if blob.centroid[0] >= 40
        ]
        buttons.sort(key=lambda blob: blob.centroid[1])
        if len(buttons) != 2:
            return
        controls[color] = [
            (int(round(blob.centroid[1])), int(round(blob.centroid[0])))
            for blob in buttons
        ]

    def press(color, side, times=1):
        if env.terminal() or env.levels_completed > base_level:
            return
        x, y = controls[color][side]
        click_coordinate(env, x, y, times=times, action=action)

    # Partially extend both sliders until each meets its first moving barrier.
    press(horizontal_slider_color, 1, 6)
    press(vertical_slider_color, 1, 3)
    press(upper_limb_color, 0)
    press(vertical_slider_color, 1)

    # Retract and shift the coupled upper barrier past the vertical slider.
    press(horizontal_slider_color, 0)
    press(upper_joint_color, 1)
    press(vertical_slider_color, 1)
    press(lower_limb_color, 0, 2)
    press(horizontal_slider_color, 0)
    press(upper_joint_color, 1)
    press(horizontal_slider_color, 0)
    press(upper_joint_color, 1)
    press(upper_limb_color, 0, 4)

    # Dock the horizontal slider in the nearer hollow target.
    press(horizontal_slider_color, 1, 8)

    # Ratchet the lower barrier around the vertical slider, then dock it.
    for _ in range(4):
        press(vertical_slider_color, 0)
        press(lower_joint_color, 1)
    press(horizontal_slider_color, 1)
    press(lower_limb_color, 0, 3)
    press(vertical_slider_color, 1, 7)
