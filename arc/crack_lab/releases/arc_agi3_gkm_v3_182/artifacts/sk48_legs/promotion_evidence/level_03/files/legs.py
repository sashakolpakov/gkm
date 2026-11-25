"""Reusable skills inferred from public frames and clone experiments."""
from perception import arr, connected_components, safe_step


def collect_ordered_chain(env, clearance=None):
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
        target = board_blocks(marker.color)[0]
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
        while board_blocks(colors[0])[0].bbox[1] > head.bbox[3] + 2:
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
