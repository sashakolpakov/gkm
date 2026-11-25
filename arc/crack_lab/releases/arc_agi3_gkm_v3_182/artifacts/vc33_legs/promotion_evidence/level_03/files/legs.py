"""Small reusable skills based only on public frame observations."""
from perception import connected_components, safe_step


def align_horizontal_indicator(env, max_moves=16):
    """Use paired vertical buttons to align a movable marker with a rail mark.

    Clone observations: the upper color-9 button moves the color-11 indicator
    right by four pixels; the lower button moves it left. The rail's color-11
    mark is four pixels high, while the moving indicator is six pixels high.
    Absolute horizontal separation supplies progress before the level reward.
    """
    initial_level = env.levels_completed
    for _ in range(max_moves):
        if env.terminal() or env.levels_completed > initial_level:
            return
        blobs = connected_components(env.frame())
        marks = [b for b in blobs if b.color == 11]
        buttons = sorted((b for b in blobs if b.color == 9),
                         key=lambda b: b.centroid[0])
        rail_marks = [b for b in marks if b.size[0] == 4]
        indicators = [b for b in marks if b.size[0] == 6]
        if len(buttons) != 2 or len(rail_marks) != 1 or len(indicators) != 1:
            return
        separation = indicators[0].bbox[1] - rail_marks[0].bbox[1]
        if separation == 0:
            return
        button = buttons[1 if separation > 0 else 0]
        y, x = button.centroid
        safe_step(env, (6, int(x), int(y)))


def align_indicator_with_transfers(env, max_moves=32):
    """Align a marker by transferring width through an intermediate reservoir.

    Four vertically ordered buttons transfer four pixels between adjacent
    regions. The bottom pair moves the six-high marker; moving it right
    requires width in the middle region, replenished by the second button.
    Marker separation and intermediate width provide observable progress.
    """
    initial_level = env.levels_completed
    for _ in range(max_moves):
        if env.terminal() or env.levels_completed > initial_level:
            return
        frame = env.frame()
        blobs = connected_components(frame)
        buttons = sorted((b for b in blobs if b.color == 9),
                         key=lambda b: b.centroid[0])
        targets = [b for b in blobs if b.color == 14 and b.size[0] == 4]
        markers = [b for b in blobs if b.color == 14 and b.size[0] == 6]
        if len(buttons) != 4 or len(targets) != 1 or len(markers) != 1:
            return
        separation = targets[0].bbox[1] - markers[0].bbox[1]
        if separation == 0:
            return
        if separation < 0:
            button = buttons[2]
        else:
            middle_row = int((buttons[1].centroid[0] +
                              buttons[2].centroid[0]) / 2)
            stock = sum(int(pixel) == 0 for pixel in frame[middle_row])
            button = buttons[3] if stock else buttons[1]
        y, x = button.centroid
        safe_step(env, (6, int(x), int(y)))


def align_reservoir_chain(env, max_moves=64):
    """Route stock through adjacent reservoirs to match colored height marks.

    Paired buttons on each boundary transfer two pixels of height. Reservoir
    capacity can block a transfer, so verify each proposed move on a clone.
    Prefix stock error measures the remaining transport distance, including
    work through reservoirs without a marker. Unmarked reservoirs finish empty.
    """
    initial_level = env.levels_completed
    blobs = connected_components(env.frame())
    buttons = sorted((b for b in blobs if b.color == 9),
                     key=lambda b: b.centroid[1])
    if not buttons or len(buttons) % 2:
        return
    boundaries = [int((buttons[i].centroid[1] +
                       buttons[i + 1].centroid[1]) / 2)
                  for i in range(0, len(buttons), 2)]
    edges = [0] + boundaries + [len(env.frame()[0])]
    bottom = max(b.bbox[2] for b in buttons)
    targets = [0] * (len(boundaries) + 1)
    for marker in blobs:
        if marker.size != (1, 3):
            continue
        marks = [b for b in blobs if b.color == marker.color
                 and b.size == (1, 2)]
        if len(marks) != 1:
            continue
        tank = sum(marker.centroid[1] >= edge for edge in boundaries)
        targets[tank] = bottom - marks[0].bbox[0]

    def heights(arena):
        frame = arena.frame()
        return [bottom + 1 - min(
            (r for r in range(bottom + 1)
             if any(int(frame[r][x]) == 0 for x in range(left, right))),
            default=bottom + 1)
            for left, right in zip(edges, edges[1:])]

    def distance(stock):
        return sum(abs(sum(stock[:i]) - sum(targets[:i]))
                   for i in range(1, len(stock)))

    for _ in range(max_moves):
        if env.terminal() or env.levels_completed > initial_level:
            return
        stock = heights(env)
        score = distance(stock)
        if not score or sum(stock) != sum(targets):
            return
        for i in range(len(boundaries)):
            deficit = sum(targets[:i + 1]) - sum(stock[:i + 1])
            if not deficit:
                continue
            source = i + 1 if deficit > 0 else i
            if not stock[source]:
                continue
            button = buttons[2 * i + (deficit < 0)]
            y, x = button.centroid
            action = (6, int(x), int(y))
            clone = env.clone()
            safe_step(clone, action)
            if (clone.levels_completed > initial_level or
                    distance(heights(clone)) < score):
                safe_step(env, action)
                break
        else:
            return
