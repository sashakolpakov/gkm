"""Small reusable skills derived from public frame observations."""
from perception import arr, connected_components, safe_step


def match_colour_row(env):
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
