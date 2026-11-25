"""Reusable skills inferred from public frames and clone experiments."""
from perception import arr, connected_components, safe_step


def align_marked_bar_tips(env, max_moves=40):
    """Click paired controls to bring coloured bar tips onto cross marks.

    Crosses are four pink pixels around an empty centre; the moving tips
    are pink too. Choose controls by their measured reduction in distance.
    """
    frame = arr(env.frame())
    marks = []
    for y in range(1, 63):
        for x in range(1, 63):
            if frame[y, x] == 5 and all(
                frame[y + dy, x + dx] == 13
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))
            ):
                marks.append((x, y))
    controls = []
    for blob in connected_components(frame, colors=[4], min_area=4):
        y0, x0, y1, x1 = blob.bbox
        controls.append((6, (x0 + x1) // 2, (y0 + y1) // 2))
    if not marks or not controls:
        return
    cross_pixels = {
        (x + dx, y + dy)
        for x, y in marks
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))
    }

    def distance(candidate):
        pixels = arr(candidate.frame())
        ys, xs = (pixels == 13).nonzero()
        tips = [(int(x), int(y)) for y, x in zip(ys, xs)
                if (int(x), int(y)) not in cross_pixels]
        # A tip must serve its own mark; nearest-tip sums can strand a
        # distant tip when another tip is close to both marks.
        costs = {0: 0}
        for x, y in marks:
            following = {}
            for mask, cost in costs.items():
                for i, (tx, ty) in enumerate(tips):
                    if not mask & (1 << i):
                        key = mask | (1 << i)
                        value = cost + abs(x - tx) + abs(y - ty)
                        following[key] = min(following.get(key, 10000), value)
            costs = following
        return min(costs.values(), default=10000)

    level = env.levels_completed
    for _ in range(max_moves):
        if env.terminal() or env.levels_completed > level:
            return

        best_action = None
        best_score = distance(env)
        for action in controls:
            clone = env.clone()
            safe_step(clone, action)
            if clone.levels_completed > level:
                best_action = action
                break
            score = distance(clone)
            if score < best_score:
                best_action, best_score = action, score
        if best_action is None:
            return
        safe_step(env, best_action)


def park_crossing_telescopic_barriers(env, max_moves=60, stages=None):
    """Park coupled horizontal/vertical barriers outside crossing tip lanes.

    First separate the carriers sideways and upward, then retract the
    perpendicular carriers. Saturate each joint using observed geometry,
    verifying the complete parking sequence on a clone before committing.
    Optional (colour, side, click_limit) stages permit separated parking lanes.
    """
    import numpy as np

    frame = arr(env.frame())
    controls = {}
    buttons = connected_components(frame, colors=[4], min_area=4)
    if not buttons:
        return
    ui_top = min(b.bbox[0] for b in buttons) - 1
    for blob in buttons:
        y0, x0, y1, x1 = blob.bbox
        glyph = frame[y0:y1 + 1, x0:x1 + 1]
        colors, counts = np.unique(glyph[glyph != 4], return_counts=True)
        if len(colors):
            color = int(colors[counts.argmax()])
            controls.setdefault(color, []).append((6, (x0+x1)//2, (y0+y1)//2))
    if stages is None:
        stages = ((9, 1, 16), (14, 1, 16), (8, 0, 16), (12, 0, 16))
    if any(len(controls.get(color, [])) != 2 for color, _, _ in stages):
        return
    clone = env.clone()
    path = []
    for color, side, click_limit in stages:
        action = sorted(controls[color], key=lambda a: a[1])[side]
        for _ in range(click_limit):
            before = arr(clone.frame())[:ui_top].copy()
            safe_step(clone, action)
            if np.array_equal(before, arr(clone.frame())[:ui_top]):
                break
            path.append(action)
            if len(path) > max_moves:
                return
    for action in path:
        safe_step(env, action)


def route_telescoping_arm_through_gap(env, max_moves=100):
    """Route a right/up/right/down arm via clear, grid-aligned bends.

    Derive candidate bends from the visible arm and obstacles. Verify each
    complete staged extension on a clone before committing its control clicks.
    """
    from perception import replay

    frame = arr(env.frame())
    controls = []
    for blob in connected_components(frame, colors=[4], min_area=4):
        y0, x0, y1, x1 = blob.bbox
        controls.append((6, (x0 + x1) // 2, (y0 + y1) // 2))
    controls.sort(key=lambda action: action[1])
    if len(controls) != 8:
        return
    ui_top = min(action[2] for action in controls) - 3
    parts = {}
    for blob in connected_components(frame, colors=[12, 10, 11, 14]):
        if blob.bbox[2] < ui_top:
            parts[blob.color] = blob.bbox
    if len(parts) != 4:
        return
    marks = []
    cross_pixels = set()
    for y in range(1, ui_top):
        for x in range(1, 63):
            ring = [(x + dx, y + dy)
                    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))]
            if frame[y, x] == 5 and all(frame[py, px] == 13 for px, py in ring):
                marks.append((x, y))
                cross_pixels.update(ring)
    if len(marks) != 1:
        return
    tips = [(int(x), int(y)) for y, x in zip(*(frame == 13).nonzero())
            if y < ui_top and (int(x), int(y)) not in cross_pixels]
    if len(tips) != 1:
        return
    target_x, target_y = marks[0]
    tip_x, tip_y = tips[0]
    base_y = (parts[12][0] + parts[12][2]) // 2
    base_x = parts[12][1] - 1
    joint_x = (parts[10][1] + parts[10][3]) // 2
    joint_y = (parts[11][0] + parts[11][2]) // 2
    unit = parts[10][3] - parts[10][1] + 1
    radius = unit // 2

    def clear(x0, y0, x1, y1):
        return (0 <= x0 <= x1 < 64 and 0 <= y0 <= y1 < ui_top
                and not (frame[y0:y1 + 1, x0:x1 + 1] == 15).any())

    candidates = []
    for bend_x in range(joint_x, target_x, unit):
        for bend_y in range(joint_y, 0, -unit):
            shifted_tip_x = tip_x + bend_x - joint_x
            shifted_tip_y = tip_y + bend_y - joint_y
            changes = (bend_x - joint_x, joint_y - bend_y,
                       target_x - shifted_tip_x, target_y - shifted_tip_y)
            if any(value < 0 or value % unit for value in changes):
                continue
            if not all((
                clear(base_x, base_y - radius, bend_x + radius, base_y + radius),
                clear(bend_x - radius, bend_y - radius, bend_x + radius, base_y + radius),
                clear(bend_x - radius, bend_y - radius, target_x + radius, bend_y + radius),
                clear(target_x - radius, bend_y - radius, target_x + radius, target_y + radius),
            )):
                continue
            path = [controls[2 * i + 1] for i, delta in enumerate(changes)
                    for _ in range(delta // unit)]
            if len(path) <= max_moves:
                candidates.append(path)
    level = env.levels_completed
    for path in sorted(candidates, key=len)[:12]:
        clone = replay(env, path)
        if clone.levels_completed > level:
            for action in path:
                if env.terminal() or env.levels_completed > level:
                    return
                safe_step(env, action)
            return
