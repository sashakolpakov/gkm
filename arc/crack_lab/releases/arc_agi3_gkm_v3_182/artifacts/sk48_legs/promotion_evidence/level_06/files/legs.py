"""Reusable skills inferred from public frames and clone experiments."""
from perception import arr, connected_components, safe_step


def collect_ordered_chain(env, clearance=None, lowest_first=False):
    """Extend a row-mounted gripper to collect the colors shown in its order strip.

    Move the retracted chain between rows, then extend until the next order
    marker fills. Plan and verify on a clone before replaying on the real env.
    An optional clearance leg returns vertical moves needed before pulling.
    """
    trial = env.clone()
    base_level = trial.levels_completed
    blobs = connected_components(trial.frame(), min_area=12)
    heads = [b for b in blobs if b.color == 6 and b.size == (6, 6)]
    head = min(heads, key=lambda b: b.bbox[0])
    strip_head = max(heads, key=lambda b: b.bbox[0])
    strip_y = strip_head.centroid[0]
    markers = sorted(
        [b for b in blobs if b.area == 16 and b.bbox[0] <= strip_y <= b.bbox[2]
         and b.bbox[1] > strip_head.bbox[3]],
        key=lambda b: b.bbox[1],
    )
    colors = [b.color for b in markers]
    path = []

    def step(action):
        if len(path) >= 100:
            raise RuntimeError('Ordered chain planning exceeded its move bound')
        safe_step(trial, action)
        path.append(action)

    def board_blocks(color):
        return [b for b in connected_components(trial.frame(), colors=[color], min_area=12)
                if b.bbox[2] < strip_head.bbox[0]]

    for marker in markers:
        targets = board_blocks(marker.color)
        target = max(targets, key=lambda b: (b.bbox[0], b.bbox[1])) if lowest_first else targets[0]
        while head.bbox[0] != target.bbox[0] - 1:
            step(1 if head.bbox[0] > target.bbox[0] - 1 else 2)
            head = min(
                [b for b in connected_components(trial.frame(), colors=[6], min_area=12)
                 if b.size == (6, 6)], key=lambda b: b.bbox[0])
        my, mx = (int(v) for v in marker.centroid)
        while int(arr(trial.frame())[my, mx]) != 0:
            step(4)
            if trial.levels_completed > base_level:
                break
        if trial.levels_completed > base_level:
            break
        # The first collected block must be adjacent to the gripper before
        # changing rows; one more retraction would release a chain member.
        if clearance is not None:
            for action in clearance(trial):
                step(action)
            head = min(
                [b for b in connected_components(trial.frame(), colors=[6], min_area=12)
                 if b.size == (6, 6)], key=lambda b: b.bbox[0])
        while min(b.bbox[1] for b in board_blocks(colors[0])
                  if b.bbox[0] == head.bbox[0] + 1) > head.bbox[3] + 2:
            step(3)
    if trial.levels_completed <= base_level:
        raise RuntimeError('Ordered chain plan did not complete the level')
    for action in path:
        if env.terminal() or env.levels_completed > base_level:
            break
        safe_step(env, action)


def stage_chain_blockers(env):
    """Park obstructing blocks on empty rows, with room for an ordered chain.

    A left action with no remaining extension releases the tail. Subsequent
    right actions push that released block without attaching it. Retract the
    empty arm before changing rows, leaving the parked block accessible later.
    """
    trial = env.clone()
    blobs = connected_components(trial.frame(), min_area=12)
    heads = sorted([b for b in blobs if b.color == 6 and b.size == (6, 6)],
                   key=lambda b: b.bbox[0])
    strip = heads[-1]
    markers = sorted([b for b in blobs if b.area == 16
                      and b.bbox[0] <= strip.centroid[0] <= b.bbox[2]
                      and b.bbox[1] > strip.bbox[3]], key=lambda b: b.bbox[1])
    board = max([b for b in blobs if b.color == 4 and b.bbox[2] < strip.bbox[0]],
                key=lambda b: b.area)
    pitch = heads[0].size[0]
    first_x = heads[0].bbox[3] + 2
    path = []

    def head():
        return min([b for b in connected_components(trial.frame(), colors=[6], min_area=12)
                    if b.size == (6, 6)], key=lambda b: b.bbox[0])

    def blocks():
        return [b for b in connected_components(trial.frame(), min_area=12)
                if b.area == 16 and b.bbox[2] < strip.bbox[0]
                and b.color in [m.color for m in markers]]

    def block(color):
        return next(b for b in blocks() if b.color == color)

    def step(action):
        if len(path) >= 160:
            raise RuntimeError('Block staging exceeded its move bound')
        safe_step(trial, action)
        path.append(action)

    target = block(markers[0].color)
    blockers = sorted([b for b in blocks() if b.bbox[0] == target.bbox[0]
                       and b.bbox[1] < target.bbox[1]], key=lambda b: b.bbox[1])
    empty_rows = [y for y in range(board.bbox[0] + 1, board.bbox[2], pitch)
                  if all(b.bbox[0] != y for b in blocks())]
    if len(empty_rows) < len(blockers):
        raise RuntimeError('Insufficient empty staging rows')
    for index, obstacle in enumerate(blockers):
        color = obstacle.color
        while head().bbox[0] != block(color).bbox[0] - 1:
            step(1 if head().bbox[0] > block(color).bbox[0] - 1 else 2)
        for _ in range(10):
            test = trial.clone()
            safe_step(test, 1)
            moved = next(b for b in connected_components(test.frame(), colors=[color], min_area=12)
                         if b.area == 16 and b.bbox[2] < strip.bbox[0])
            if moved.bbox[0] != block(color).bbox[0]:
                break
            step(4)
        else:
            raise RuntimeError('Could not attach staging block')
        destination = empty_rows[index]
        while block(color).bbox[0] != destination:
            step(1 if block(color).bbox[0] > destination else 2)
        while block(color).bbox[1] > first_x:
            step(3)
        step(3)  # release the single attached block
        parking_x = board.bbox[3] - 10 - index * pitch
        while block(color).bbox[1] < parking_x:
            step(4)
        for _ in range(10):
            before = arr(trial.frame()).copy()
            step(3)
            if (arr(trial.frame()) == before).all():
                break
    # Verify both staging and its intended continuation before committing.
    validation = trial.clone()
    collect_ordered_chain(validation)
    if validation.levels_completed <= env.levels_completed:
        raise RuntimeError('Staging did not enable ordered collection')
    for action in path:
        safe_step(env, action)


def clear_chain_guide(env):
    """Find a short vertical escape from a guide that prevents chain pulling.

    Return a clone-verified path, preserving the filled order prefix. A pull
    must shift the first carried block left without releasing an order marker.
    """
    from collections import deque

    blobs = connected_components(env.frame(), min_area=12)
    heads = sorted([b for b in blobs if b.color == 6 and b.size == (6, 6)],
                   key=lambda b: b.bbox[0])
    strip = heads[-1]
    # Filled markers are hollow and therefore absent from the area-16 list.
    f = arr(env.frame())
    strip_y = int(strip.centroid[0])
    filled = [(strip_y, x) for x in range(strip.bbox[3] + 3, 63, 6)
              if int(f[strip_y, x]) == 0]
    first_color = int(f[strip_y, strip.bbox[3] + 2])
    first = next(b for b in blobs if b.color == first_color and b.area == 16
                 and b.bbox[2] < strip.bbox[0])
    if first.bbox[1] <= heads[0].bbox[3] + 2:
        return []

    def pullable(state):
        test = state.clone()
        safe_step(test, 3)
        frame = arr(test.frame())
        blocks = [b for b in connected_components(frame, colors=[first.color], min_area=12)
                  if b.area == 16 and b.bbox[2] < strip.bbox[0]]
        return (blocks and blocks[0].bbox[1] < first.bbox[1]
                and all(int(frame[y, x]) == 0 for y, x in filled))

    queue = deque([(env.clone(), [])])
    seen = set()
    while queue and len(seen) < 40:
        state, path = queue.popleft()
        key = arr(state.frame()).tobytes()
        if key in seen:
            continue
        seen.add(key)
        if pullable(state):
            return path
        if len(path) >= 8:
            continue
        for action in (1, 2):
            child = state.clone()
            safe_step(child, action)
            queue.append((child, path + [action]))
    raise RuntimeError('No vertical guide escape preserves the carried chain')


def distribute_chain_to_rails(env, beam_width=100, max_depth=90):
    """Distribute linked cargo into the vertical orders shown beside rail IDs.

    Pulling can release a tail; rails constrain horizontal motion. Detached
    cargo can be pushed vertically into a rail from below. Search public-frame
    clones using rail distance, vertical order, and filled order markers, then
    commit only a plan that increases the real level reward.
    """
    import time

    initial = arr(env.frame())
    blobs = connected_components(initial, min_area=4)
    cargo = [b for b in blobs if b.area == 16]
    strip_y = max(b.bbox[0] for b in cargo)
    markers = sorted([b for b in cargo if b.bbox[0] == strip_y],
                     key=lambda b: b.bbox[1])
    board = max([b for b in blobs if b.color == 4 and b.bbox[2] < strip_y],
                key=lambda b: b.area)
    icons = sorted([b for b in blobs if b.area == 4
                    and b.bbox[2] < board.bbox[0]], key=lambda b: b.bbox[1])
    labels = sorted([b for b in blobs if b.area == 4 and b.bbox[0] > strip_y
                     and b.color in {i.color for i in icons}],
                    key=lambda b: b.bbox[1])
    pitch = next(b.size[0] for b in blobs if b.color == 6 and b.size == (6, 6))
    groups = []
    for index, label in enumerate(labels):
        right = labels[index + 1].bbox[1] if index + 1 < len(labels) else initial.shape[1]
        order = [m for m in markers if label.bbox[3] < m.bbox[1] < right]
        icon = next(b for b in icons if b.color == label.color)
        end = board.bbox[0]
        while end <= board.bbox[2] and all(
                int(initial[end, x]) in (2, 3)
                for x in range(icon.bbox[1], icon.bbox[3] + 1)):
            end += 1
        groups.append((order, icon.bbox[1] - 1, end - order[0].size[0]))
    if not groups or sum(len(g[0]) for g in groups) != len(markers):
        raise RuntimeError('Could not read vertical rail orders')
    colors = {m.color for m in markers}
    if len(colors) != len(markers):
        raise RuntimeError('Rail search requires distinct cargo colors')

    def score(frame):
        positions = {}
        for color in colors:
            ys, xs = (frame[:board.bbox[2] + 1] == color).nonzero()
            positions[color] = (int(ys.min()), int(xs.min()))
        cost = 0
        for order, x, bottom in groups:
            for marker in order:
                y, bx = positions[marker.color]
                cost += 2 * abs(bx - x) / pitch + max(0, y - bottom) / pitch
            for first, second in zip(order, order[1:]):
                cost += 2 * max(0, positions[first.color][0] + pitch
                                - positions[second.color][0]) / pitch
        filled = sum(int(frame[int(m.centroid[0]), int(m.centroid[1])]) == 0
                     for m in markers)
        return cost - 4 * filled

    base_level = env.levels_completed
    beam = [(score(initial), env.clone(), [])]
    seen = {initial.tobytes()}
    started, steps = time.monotonic(), 0
    for _ in range(max_depth):
        candidates = []
        for _, state, path in beam:
            for action in (1, 2, 3, 4):
                child = state.clone()
                safe_step(child, action)
                steps += 1
                next_path = path + [action]
                if child.levels_completed > base_level:
                    for move in next_path:
                        if env.terminal() or env.levels_completed > base_level:
                            break
                        safe_step(env, move)
                    return
                frame = arr(child.frame())
                key = frame.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                candidates.append((score(frame), child, next_path))
        if not candidates:
            break
        candidates.sort(key=lambda item: item[0])
        beam = candidates[:beam_width]
        # Keep bounded foreground clone work below roughly 300 steps/second.
        delay = steps / 280 - (time.monotonic() - started)
        if delay > 0:
            time.sleep(delay)
    raise RuntimeError('Rail distribution search exhausted its bounds')


def relay_chain_around_barrier(env):
    """Use released cargo as vertical push relays around a shaft barrier.

    Split a three-block chain over two upper rows, push the released pieces
    beyond the barrier column, then sweep down from above. Recursive pushes
    expose two far-side blocks below the barrier. Verify ordered collection
    before committing the staging moves.
    """
    trial = env.clone()
    blobs = connected_components(trial.frame(), min_area=12)
    head = min((b for b in blobs if b.color == 6 and b.size == (6, 6)),
               key=lambda b: b.bbox[0])
    pitch = head.size[0]
    board = max((b for b in blobs if b.color == 4 and b.bbox[2] < 50),
                key=lambda b: b.area)
    barriers = [b for b in blobs if b.color == 5 and b.size == (pitch, pitch)
                and b.bbox[0] == head.bbox[0]]
    if len(barriers) != 1:
        raise RuntimeError('Relay staging requires one isolated shaft barrier')
    barrier = barriers[0]
    beyond = (barrier.bbox[1] - board.bbox[1]) // pitch + 1
    width = board.size[1] // pitch
    if beyond != 4 or width < beyond + 2 or head.bbox[0] - board.bbox[0] < 3 * pitch:
        raise RuntimeError('Insufficient relay staging clearance')
    path = ([1, 3, 1, 3, 3, 2] + [4] * beyond + [3] * beyond
            + [1] + [4] * beyond + [3] * beyond + [1]
            + [4] * (beyond + 2) + [2, 2] + [3] * (beyond + 2))
    for action in path:
        safe_step(trial, action)
    validation = trial.clone()
    collect_ordered_chain(validation, lowest_first=True)
    if validation.levels_completed <= env.levels_completed:
        raise RuntimeError('Relay staging did not expose an ordered solution')
    for action in path:
        safe_step(env, action)


def exchange_cargo_between_shafts(env):
    """Hand crossed cargo orders between perpendicular, selectable grippers.

    Stage the column cargo against the left edge. The top gripper pushes
    the row cargo one cell below the central obstruction, then collects the
    staged column. The left gripper can now collect its unobstructed row.
    Boundary contact attaches pushed cargo; retracting beyond the origin
    releases it. Verify the whole exchange on a public clone before replay.
    """
    blobs = connected_components(env.frame(), min_area=12)
    board = max((b for b in blobs if b.color == 4 and b.bbox[2] < 50),
                key=lambda b: b.area)
    horizontal = min((b for b in blobs if b.color == 6 and b.size == (6, 6)),
                     key=lambda b: b.bbox[0])
    vertical = min((b for b in blobs if b.color == 15 and b.size == (6, 6)),
                   key=lambda b: b.bbox[0])
    pitch = horizontal.size[0]
    height, width = (n // pitch for n in board.size)
    blocks = [b for b in blobs if b.area == 16
              and board.bbox[0] <= b.bbox[0] <= board.bbox[2]]
    row_blocks = sorted((b for b in blocks if b.bbox[0] == horizontal.bbox[0] + 1),
                        key=lambda b: b.bbox[1])
    column_blocks = sorted((b for b in blocks if b.bbox[1] == vertical.bbox[1] + 1),
                           key=lambda b: b.bbox[0])
    row = (horizontal.bbox[0] - board.bbox[0]) // pitch
    column = (vertical.bbox[1] - board.bbox[1]) // pitch
    if (not row_blocks or not column_blocks
            or len({b.color for b in row_blocks}) != 1
            or len({b.color for b in column_blocks}) != 1
            or any((b.bbox[0] - board.bbox[0]) // pitch <= row
                   for b in column_blocks)):
        raise RuntimeError('Crossed shaft exchange requires separated row and column cargo')
    path = []

    def move_to(current, target, negative, positive):
        path.extend([negative if target < current else positive] * abs(target - current))
        return target

    current_row = row
    for block in column_blocks:
        target_row = (block.bbox[0] - board.bbox[0]) // pitch
        current_row = move_to(current_row, target_row, 1, 2)
        path.extend([4] * (width + 1) + [3] * (width + 1))
    move_to(current_row, row, 1, 2)
    path.append((6, int(vertical.centroid[1]), int(vertical.centroid[0])))
    path.extend([1] * height)
    current_column = column
    for block in row_blocks:
        target_column = (block.bbox[1] - board.bbox[1]) // pitch
        current_column = move_to(current_column, target_column, 3, 4)
        path.extend([2] * (row + 1) + [1] * (row + 1))
    move_to(current_column, 0, 3, 4)
    path.extend([2] * (height + len(column_blocks)))
    path.extend([1] * (height - len(column_blocks)))
    path.append((6, int(horizontal.centroid[1]), int(horizontal.centroid[0])))
    path.extend([2] + [4] * (width + 2))

    trial = env.clone()
    base_level = env.levels_completed
    verified = []
    for action in path:
        safe_step(trial, action)
        verified.append(action)
        if trial.levels_completed > base_level:
            break
    if trial.levels_completed <= base_level:
        raise RuntimeError('Crossed shaft exchange did not complete the level')
    for action in verified:
        if env.terminal() or env.levels_completed > base_level:
            break
        safe_step(env, action)
