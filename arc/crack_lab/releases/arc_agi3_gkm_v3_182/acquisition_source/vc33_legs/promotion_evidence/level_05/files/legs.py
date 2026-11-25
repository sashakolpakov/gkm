"""Small reusable skills based only on public frame observations."""
from perception import connected_components, safe_step
from collections import deque


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


def float_through_locks(env, max_states=4000):
    """Equalize lock levels, cross, then lift a floating figure to its mark.

    Observed buttons move three rows of liquid between neighboring tanks.
    A lock becomes traversable at equal surfaces at its lower edge; clicking
    it moves the figure to the neighboring tank. The figure rides the surface,
    and reaching the two-row goal mark completes the level automatically.
    Search liquid heights and figure compartment, ignoring the top-row timer.
    Verify the entire symbolic plan on a clone before committing it.
    """
    blobs = connected_components(env.frame())
    locks = sorted((b for b in blobs if b.color == 1 and b.size[1] == 3),
                   key=lambda b: b.centroid[1])
    walls = sorted({int(b.centroid[1]) for b in blobs
                    if b.color in (1, 5, 11) and b.size[1] == 3},
                   key=int)
    goals = [b for b in blobs if b.color == 11 and b.size == (2, 3)]
    rafts = [b for b in blobs if b.color == 11 and b.size == (2, 6)]
    buttons = [b for b in blobs if b.color == 9]
    if not locks or len(goals) != 1 or len(rafts) != 1 or not buttons:
        return
    frame = env.frame()
    bottom = len(frame)
    edges = [0] + [x + 2 for x in walls] + [len(frame[0])]
    samples = [(a + b - 1) // 2 for a, b in zip(edges, edges[1:])]
    stock = tuple(bottom - next((y for y in range(1, bottom)
                                if int(frame[y][x]) == 0), bottom)
                  for x in samples)
    position = sum(rafts[0].centroid[1] > x for x in walls)
    goal_wall = walls.index(int(goals[0].centroid[1]))
    goal_height = bottom - goals[0].bbox[2] - 1
    transfers = []
    for button in buttons:
        y, x = button.centroid
        boundary = min(range(len(walls)), key=lambda i: abs(walls[i] - x))
        left, right = boundary, boundary + 1
        source, dest = (right, left) if x < walls[boundary] else (left, right)
        transfers.append((source, dest, (6, int(x), int(y))))
    crossings = [(walls.index(int(b.centroid[1])), bottom - b.bbox[2] - 1,
                  (6, int(b.centroid[1]), int(b.centroid[0]))) for b in locks]
    start = (stock, position)
    queue = deque([start])
    parents = {start: None}
    finish = None
    while queue and len(parents) <= max_states:
        state = queue.popleft()
        heights, tank = state
        if tank in (goal_wall, goal_wall + 1) and heights[tank] == goal_height:
            finish = state
            break
        successors = []
        for source, dest, action in transfers:
            if heights[source] < 3 or heights[dest] + 3 >= bottom:
                continue
            changed = list(heights)
            changed[source] -= 3
            changed[dest] += 3
            successors.append(((tuple(changed), tank), action))
        for boundary, height, action in crossings:
            if (tank in (boundary, boundary + 1) and
                    heights[boundary] == heights[boundary + 1] == height):
                successors.append(((heights, 2 * boundary + 1 - tank), action))
        for successor, action in successors:
            if successor not in parents:
                parents[successor] = (state, action)
                queue.append(successor)
    if finish is None:
        return
    path = []
    while parents[finish] is not None:
        finish, action = parents[finish]
        path.append(action)
    path.reverse()
    initial_level = env.levels_completed
    clone = env.clone()
    for action in path:
        safe_step(clone, action)
        if clone.terminal() or clone.levels_completed > initial_level:
            break
    if clone.levels_completed <= initial_level:
        return
    for action in path:
        if env.terminal() or env.levels_completed > initial_level:
            return
        safe_step(env, action)


def exchange_figures_through_locks(env, max_states=100000):
    """Route several colored figures through horizontal reservoir locks.

    Buttons transfer three columns of fill. Equal fill at a gate's right
    edge allows a click to exchange every figure in its adjacent reservoirs.
    Barrier starts bound reservoir drainage. All figures must simultaneously
    meet their matching marks. Search stock and figure locations, then verify
    the complete route on a clone before spending real moves.
    """
    blobs = connected_components(env.frame())
    gates = sorted((b for b in blobs if b.color == 1 and b.size[0] == 3),
                   key=lambda b: b.centroid[0])
    if not gates:
        return
    boundaries = [int(b.centroid[0]) for b in gates]
    buttons = sorted((b for b in blobs if b.color == 9),
                     key=lambda b: b.centroid[0])
    if len(buttons) != 2 * len(gates):
        return
    figures = sorted((b for b in blobs if b.size == (6, 2)),
                     key=lambda b: b.color)
    goals = []
    for figure in figures:
        marks = [b for b in blobs if b.color == figure.color
                 and b.size == (3, 2)]
        if len(marks) != 1:
            return
        mark = marks[0]
        goals.append((min(range(len(gates)),
                          key=lambda i: abs(boundaries[i] - mark.centroid[0])),
                      mark.bbox[3] + 1))
    if not figures:
        return
    limits = [min(b.bbox[1] for b in blobs if b.color == 5
                  and b.bbox[0] <= y <= b.bbox[2]) for y in boundaries]
    minimum = [max(limits[max(0, i - 1):min(len(limits), i + 1)])
               for i in range(len(gates) + 1)]
    edges = [1] + [b.bbox[2] + 1 for b in gates] + [64]
    ends = [b.bbox[0] for b in gates] + [64]
    rows = [(a + b - 1) // 2 for a, b in zip(edges, ends)]
    frame = env.frame()
    stock = tuple(next((x for x in range(64) if int(frame[y][x]) == 0),
                       64) for y in rows)
    positions = tuple(sum(b.centroid[0] > y for y in boundaries)
                      for b in figures)
    transfers = []
    for i, button in enumerate(buttons):
        y, x = button.centroid
        transfers.append((i // 2, -1 if i % 2 == 0 else 1,
                          (6, int(x), int(y))))
    crossings = [(b.bbox[3] + 1,
                  (6, int(b.centroid[1]), int(b.centroid[0]))) for b in gates]
    start = (stock, positions)
    parents = {start: None}
    queue = deque([start])
    finish = None
    while queue and len(parents) <= max_states:
        state = queue.popleft()
        heights, occupants = state
        if all(p in (wall, wall + 1) and heights[p] == height
               for p, (wall, height) in zip(occupants, goals)):
            finish = state
            break
        successors = []
        for i, direction, action in transfers:
            changed = list(heights)
            changed[i] += 3 * direction
            changed[i + 1] -= 3 * direction
            if all(lo <= v <= 64 for lo, v in zip(minimum, changed)):
                successors.append(((tuple(changed), occupants), action))
        for i, (height, action) in enumerate(crossings):
            if heights[i] == heights[i + 1] == height:
                moved = tuple(2 * i + 1 - p if p in (i, i + 1) else p
                              for p in occupants)
                successors.append(((heights, moved), action))
        for successor, action in successors:
            if successor not in parents:
                parents[successor] = (state, action)
                queue.append(successor)
    if finish is None:
        return
    path = []
    while parents[finish] is not None:
        finish, action = parents[finish]
        path.append(action)
    path.reverse()
    initial_level = env.levels_completed
    clone = env.clone()
    for action in path:
        if clone.terminal() or clone.levels_completed > initial_level:
            break
        safe_step(clone, action)
    if clone.levels_completed <= initial_level:
        return
    for action in path:
        if env.terminal() or env.levels_completed > initial_level:
            break
        safe_step(env, action)
