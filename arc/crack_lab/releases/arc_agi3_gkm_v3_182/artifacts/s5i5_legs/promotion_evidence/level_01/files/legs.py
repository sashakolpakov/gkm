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
        return sum(min((abs(x - tx) + abs(y - ty) for tx, ty in tips),
                       default=128) for x, y in marks)

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
