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
