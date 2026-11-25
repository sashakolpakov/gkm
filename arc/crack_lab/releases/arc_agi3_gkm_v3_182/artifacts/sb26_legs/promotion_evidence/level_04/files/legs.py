"""Small reusable skills derived from public frame observations."""
from perception import arr, connected_components, safe_step


def match_colour_row(env, socket_order=None):
    """Fill a row of sockets from a swatch palette, matching the top legend."""
    frame = arr(env.frame())
    blobs = connected_components(frame, min_area=4)
    targets = sorted(
        (b for b in blobs if b.bbox[2] < 8 and b.area == 20),
        key=lambda b: b.bbox[1],
    )
    palette = {
        b.color: (int(b.centroid[1]), int(b.centroid[0]))
        for b in blobs if b.bbox[0] > 54 and b.area == 16
    }
    sockets = sorted(
        (b for b in blobs if b.color == 2 and b.area == 4
         and 8 < b.bbox[0] < 53),
        key=lambda b: b.bbox[1],
    )
    if not targets or len(targets) != len(sockets):
        raise ValueError('Expected one socket per target swatch')
    if socket_order is not None:
        by_position = {(int(b.centroid[1]), int(b.centroid[0])): b
                       for b in sockets}
        sockets = [by_position[position] for position in socket_order]
    actions = []
    expected = []
    for target, socket in zip(targets, sockets):
        sx, sy = palette[target.color]
        x, y = int(socket.centroid[1]), int(socket.centroid[0])
        actions.extend([(6, sx, sy), (6, x, y)])
        expected.append((x, y, target.color))

    # Dense progress: every placement must add a correctly coloured socket.
    trial = env.clone()
    base_level = env.levels_completed
    for i in range(len(expected)):
        safe_step(trial, actions[2 * i])
        safe_step(trial, actions[2 * i + 1])
        observed = arr(trial.frame())
        matched = sum(int(observed[y, x]) == colour
                      for x, y, colour in expected[:i + 1])
        if matched != i + 1:
            raise ValueError('Placement failed to increase matched sockets')
    safe_step(trial, 5)
    if trial.levels_completed <= base_level:
        raise ValueError('Matching the legend did not complete the level')
    for action in actions + [5]:
        safe_step(env, action)


def match_nested_colour_rows(env):
    """Expand coloured box references, then fill their leaves in legend order."""
    frame = arr(env.frame())
    blobs = connected_components(frame, min_area=4)
    sockets = {(int(b.centroid[1]), int(b.centroid[0]))
               for b in blobs if b.color == 2 and b.area == 4
               and 8 < b.bbox[0] < 53}
    # Read outlines geometrically: a drawn connector may join a reference
    # to its box, so connected-component bounds need not equal box bounds.
    background = int(frame[8, 0])
    boxes = {}
    links = []
    for y in range(9, 49):
        for x in range(1, 60):
            colour = int(frame[y, x])
            if colour in (background, 0, 2) or frame[y, x - 1] == colour:
                continue
            end = x
            while end + 1 < 64 and frame[y, end + 1] == colour:
                end += 1
            width = end - x + 1
            if width == 4 and y + 3 < 53:
                if (all(frame[y + 3, x:end + 1] == colour)
                        and all(frame[y:y + 4, x] == colour)
                        and all(frame[y:y + 4, end] == colour)):
                    links.append((x + 1, y + 1, colour))
            if width < 6:
                continue
            for bottom in range(y + 5, 53):
                if (sum(frame[bottom, x:end + 1] == colour) >= width - 2
                        and all(frame[y:bottom + 1, x] == colour)
                        and all(frame[y:bottom + 1, end] == colour)):
                    boxes[colour] = (y, x, bottom, end)
                    break
    contents = {}
    referenced = set()
    for colour, (r0, c0, r1, c1) in boxes.items():
        leaves = [(x, y, None) for x, y in sockets
                  if c0 < x < c1 and r0 < y < r1]
        references = [(x, y, link) for x, y, link in links
                      if c0 < x < c1 and r0 < y < r1]
        contents[colour] = sorted(leaves + references)
        referenced.update(link for x, y, link in references)
    roots = set(boxes) - referenced
    if len(roots) != 1:
        raise ValueError('Expected one root container')
    order = []
    visited = set()

    def visit(colour):
        if colour in visited or colour not in contents:
            raise ValueError('Cyclic, repeated, or unknown box reference')
        visited.add(colour)
        for x, y, link in contents[colour]:
            if link is None:
                order.append((x, y))
            else:
                visit(link)

    visit(roots.pop())
    if set(order) != sockets:
        raise ValueError('Not all sockets reachable from the root box')
    match_colour_row(env, socket_order=order)


def complete_expandable_colour_rows(env):
    """Fill expression rows, choosing references around immutable colour leaves."""
    frame = arr(env.frame())
    blobs = connected_components(frame, min_area=4)
    targets = [b.color for b in sorted(
        (b for b in blobs if b.bbox[2] < 8 and b.area == 20),
        key=lambda b: b.bbox[1])]
    swatches = {b.color: b for b in blobs
                if b.bbox[0] > 54 and b.area in (12, 16)}
    boxes = {}
    for b in blobs:
        if b.color not in swatches:
            continue
        y0, x0, y1, x1 = b.bbox
        if not (8 < y0 < y1 < 53 and x1 - x0 >= 5):
            continue
        if (all(frame[y0, x0:x1 + 1] == b.color)
                and all(frame[y1, x0:x1 + 1] == b.color)
                and all(frame[y0:y1 + 1, x0] == b.color)
                and all(frame[y0:y1 + 1, x1] == b.color)):
            boxes[b.color] = b.bbox
    references = {colour for colour, b in swatches.items() if b.area == 12}
    roots = set(boxes) - references
    if len(roots) != 1 or not references <= set(boxes):
        raise ValueError('Expected a root and palette references to other rows')
    contents = {}
    for colour, (y0, x0, y1, x1) in boxes.items():
        tokens = []
        for b in blobs:
            y, x = int(b.centroid[0]), int(b.centroid[1])
            if not (x0 < x < x1 and y0 < y < y1):
                continue
            if b.color == 2 and b.area == 4:
                tokens.append((x, y, 'socket', None))
            elif b.area == 16 and b.color in swatches:
                tokens.append((x, y, 'literal', b.color))
            elif b.area == 12 and b.color in boxes:
                tokens.append((x, y, 'reference', b.color))
        contents[colour] = sorted(tokens)
    root = roots.pop()
    visits = 0

    def plan(todo, index, used, actions):
        nonlocal visits
        visits += 1
        if visits > 10000:
            raise ValueError('Expression planning bound exceeded')
        if not todo:
            return actions if index == len(targets) and used == set(boxes) else None
        if index >= len(targets):
            return None
        x, y, kind, colour = todo[0]
        rest = todo[1:]
        if kind == 'literal':
            if colour == targets[index]:
                return plan(rest, index + 1, used, actions)
            return None
        if kind == 'reference':
            if colour in used:
                return None
            return plan(contents[colour] + rest, index, used | {colour}, actions)
        choices = [targets[index]] + sorted(references - {targets[index]})
        for colour in choices:
            if colour not in swatches:
                continue
            b = swatches[colour]
            added = [(6, int(b.centroid[1]), int(b.centroid[0])), (6, x, y)]
            if colour in references:
                if colour in used:
                    continue
                result = plan(contents[colour] + rest, index,
                              used | {colour}, actions + added)
            else:
                if colour != targets[index]:
                    continue
                result = plan(rest, index + 1, used, actions + added)
            if result is not None:
                return result
        return None

    actions = plan(contents[root], 0, {root}, [])
    if actions is None:
        raise ValueError('No expression matches the legend')
    trial = env.clone()
    expected = []
    for selection, placement in zip(actions[::2], actions[1::2]):
        safe_step(trial, selection)
        safe_step(trial, placement)
        _, sx, sy = selection
        _, x, y = placement
        colour = next(colour for colour, b in swatches.items()
                      if (int(b.centroid[1]), int(b.centroid[0])) == (sx, sy))
        expected.append((x, y, colour))
        observed = arr(trial.frame())
        # Dense progress counts preserved, correctly placed token outlines.
        matched = sum(int(observed[y - 1, x - 1]) == colour
                      for x, y, colour in expected)
        if matched != len(expected):
            raise ValueError('Expression placement did not increase progress')
    safe_step(trial, 5)
    if trial.levels_completed <= env.levels_completed:
        raise ValueError('Expanded expression did not complete the level')
    for action in actions + [5]:
        safe_step(env, action)
