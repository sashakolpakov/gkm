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
