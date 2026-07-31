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


def extend_shared_marker_through_staged_crossbars(
    env, panel_color=2, action=6
):
    """Advance a shared marker while staging long and short crossbars."""
    base_level = env.levels_completed
    panels = [
        blob
        for blob in connected_components(
            env.frame(), colors={panel_color}, min_area=20
        )
        if blob.centroid[0] >= 40
    ]
    if len(panels) != 5:
        return

    panels.sort(key=lambda blob: (blob.centroid[0], blob.centroid[1]))
    center = min(panels, key=lambda blob: abs(blob.centroid[1] - 31.5))
    outer = [blob for blob in panels if blob is not center]
    outer.sort(key=lambda blob: (blob.centroid[0], blob.centroid[1]))
    top_left, top_right, bottom_left, bottom_right = outer

    def press(panel, side, times=1):
        if env.terminal() or env.levels_completed > base_level:
            return
        r0, c0, r1, c1 = panel.bbox
        x = c0 + 3 if side == 0 else c1 - 3
        y = int(round((r0 + r1) / 2))
        click_coordinate(env, x, y, times=times, action=action)

    # Approach the long crossbar, retract it below the marker, then pass it.
    press(center, 1, 3)
    press(bottom_left, 0, 7)
    press(center, 1, 4)

    # Each shared extension advances all three remaining crossbars too.
    # Ratchet them upward once per advance so the marker can pass in turn.
    for _ in range(4):
        press(top_right, 1)
        press(top_left, 1)
        press(bottom_right, 1)
        press(center, 1)


def thread_coupled_marker_through_reconfigurable_frame(
    env, panel_color=2, action=6
):
    """Stage four coupled controls to thread a marker through a moving frame."""
    base_level = env.levels_completed
    panels = [
        blob
        for blob in connected_components(
            env.frame(), colors={panel_color}, min_area=20
        )
        if blob.centroid[0] >= 50
    ]
    if len(panels) != 4:
        return
    panels.sort(key=lambda blob: blob.centroid[1])

    def press(panel_index, side, times=1):
        if env.terminal() or env.levels_completed > base_level:
            return
        r0, c0, r1, c1 = panels[panel_index].bbox
        x = c0 + 2 if side == 0 else c1 - 2
        y = int(round((r0 + r1) / 2))
        click_coordinate(env, x, y, times=times, action=action)

    # Open the first gap and lower the marker-bearing paired rods into it.
    press(0, 0)
    press(1, 1, 3)

    # Couple the rods to the moving frame, retracting the auxiliary slider
    # before carrying the joined assembly through the central obstruction.
    press(2, 1)
    press(3, 0, 2)
    press(2, 1, 3)

    # Translate the frame clear, release the coupling, and descend past it.
    press(0, 0, 4)
    press(2, 0, 3)
    press(1, 1, 6)

    # Swing the marker into the hollow target and restore the slider.
    press(2, 1, 3)
    press(3, 1, 2)


def dock_three_link_arm_through_partitioned_chamber(
    env, panel_color=2, action=6
):
    """Reorient and extend a three-link arm through a partitioned chamber."""
    base_level = env.levels_completed
    panels = connected_components(
        env.frame(), colors={panel_color}, min_area=20
    )
    coarse = sorted(
        (blob for blob in panels if blob.centroid[0] < 52),
        key=lambda blob: blob.centroid[1],
    )
    fine = sorted(
        (blob for blob in panels if blob.centroid[0] >= 52),
        key=lambda blob: blob.centroid[1],
    )
    if len(coarse) != 3 or len(fine) != 3:
        return

    def press_coarse(index, times=1):
        blob = coarse[index]
        r0, c0, r1, c1 = blob.bbox
        click_coordinate(
            env, (c0 + c1) // 2, (r0 + r1) // 2,
            times=times, action=action,
        )

    def press_fine(index, side, times=1):
        blob = fine[index]
        r0, c0, r1, c1 = blob.bbox
        x = c0 + 3 if side == 0 else c1 - 3
        click_coordinate(
            env, x, (r0 + r1) // 2, times=times, action=action
        )

    # Fold the links into the upper-left opening, then translate the whole
    # assembly to the chamber's lower side.
    press_coarse(1, 2)
    press_coarse(2)
    press_fine(2, 1, 6)

    # Face the marker toward the target and extend it through the partition.
    press_coarse(0)
    press_fine(1, 1, 2)
    press_coarse(0)
    press_fine(0, 1, 11)
    press_fine(1, 1)


def dock_four_link_arm_through_dual_opening_chamber(
    env, panel_color=2, action=6
):
    """Thread a four-link arm around two walls while clearing a coupled gate."""
    panels = connected_components(
        env.frame(), colors={panel_color}, min_area=10
    )
    upper = sorted(
        (blob for blob in panels if blob.centroid[0] < 55),
        key=lambda blob: blob.centroid[1],
    )
    lower = sorted(
        (blob for blob in panels if blob.centroid[0] >= 55),
        key=lambda blob: blob.centroid[1],
    )
    if len(upper) != 5 or len(lower) != 4:
        return

    controls = {
        "A": upper[0], "B": upper[1], "C": upper[2],
        "D": upper[3], "E": upper[4],
        "F": lower[0], "G": lower[1], "H": lower[2],
        "I": lower[3],
    }

    def press(name, side=None, times=1):
        blob = controls[name]
        r0, c0, r1, c1 = blob.bbox
        if side is None:
            x = (c0 + c1) // 2
        else:
            x = c0 + 2 if side == 0 else c1 - 2
        click_coordinate(
            env, x, (r0 + r1) // 2, times=times, action=action
        )

    # Fold the arm below the central wall and point its terminal rail west.
    press("F", 0)
    press("A", 1)
    press("B", times=2)
    press("A", 1)
    press("F", 1)
    press("A", 1)
    press("C", 1)
    press("A", 1)
    press("C", 1)
    press("A", 1)
    press("D")
    press("H", 1, 5)

    # Compact and restage the middle links into the bottom corridor.
    press("H", 0, 5)
    press("C", 0, 3)
    press("F", 1, 4)
    press("D", times=2)
    press("C", 1, 4)
    press("A", 1, 3)
    press("D")
    press("C", 1, 3)

    # Send the tip through the left opening, then bring its parent underneath.
    press("H", 1, 8)
    press("H", 0, 5)
    press("F", 1, 2)
    for _ in range(6):
        press("C", 0)
        press("F", 1)

    # Rotate the coupled upper gate out of the final lane without undocking it.
    press("E", 0, 2)
    press("I", times=2)
    press("E", 1, 2)

    # Turn the compact child rail north, raise it, and dock the terminal marker.
    press("C", 0)
    press("H", 0, 3)
    press("D", times=2)
    press("C", 1, 7)
    press("H", 1, 6)


def extend_terminal_marker_through_coupled_crossbars(
    env, panel_color=2, action=6
):
    """Clear two coupled crossbars, then extend a terminal marker through."""
    panels = connected_components(
        env.frame(), colors={panel_color}, min_area=10
    )
    upper = sorted(
        (blob for blob in panels if blob.centroid[0] < 15),
        key=lambda blob: (blob.centroid[0], blob.centroid[1]),
    )
    central = [
        blob for blob in panels
        if 15 <= blob.centroid[0] < 25
    ]
    lower = sorted(
        (blob for blob in panels if blob.centroid[0] >= 50),
        key=lambda blob: blob.centroid[1],
    )
    if len(upper) != 4 or len(central) != 1 or len(lower) != 2:
        return

    controls = {
        "upper_gate": upper[0],
        "lower_gate": upper[2],
        "terminal": upper[3],
        "rotate": central[0],
        "lower_slide": lower[0],
        "upper_slide": lower[1],
    }

    def press(name, side=None, times=1):
        blob = controls[name]
        r0, c0, r1, c1 = blob.bbox
        x = (c0 + c1) // 2
        if side is not None:
            x = c0 + 2 if side == 0 else c1 - 2
        click_coordinate(
            env, x, (r0 + r1) // 2, times=times, action=action
        )

    # Point the upper gate's coupled arm through the chamber opening. Raise
    # the crossbar clear of its peg, retract it past the marker lane, and
    # restore the coupled arm so the central mechanism can rotate again.
    press("rotate", times=3)
    press("upper_gate", 1, 6)
    press("upper_slide", 0, 8)
    press("upper_gate", 0, 6)

    # Repeat the same maneuver for the lower crossbar from the other side.
    press("rotate", times=3)
    press("lower_gate", 1, 6)
    press("lower_slide", 0, 8)
    press("lower_gate", 0, 6)

    # Reorient the terminal arm into the opening and dock its marker.
    press("rotate", times=3)
    press("terminal", 1, 20)
