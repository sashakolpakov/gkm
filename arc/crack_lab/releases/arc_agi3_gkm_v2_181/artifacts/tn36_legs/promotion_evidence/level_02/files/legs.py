# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.

from perception import connected_components


def make_small_segments_color_5_and_submit(env):
    """Toggle each small color-1 line segment, then click the color-9 disc."""
    segments = [
        blob
        for blob in connected_components(env.frame(), colors=(1,), min_area=3)
        if blob.area == 3 and 1 in blob.size
    ]
    for blob in segments:
        row, col = blob.centroid
        env.step(6, int(round(col)), int(round(row)))

    submit_discs = [
        blob
        for blob in connected_components(env.frame(), colors=(9,), min_area=4)
        if blob.size[0] > 1 and blob.size[1] > 1
    ]
    if submit_discs:
        disc = max(submit_discs, key=lambda blob: blob.area)
        row, col = disc.centroid
        env.step(6, int(round(col)), int(round(row)))


def turn_on_outer_rows_of_right_segment_panel_and_submit(env):
    """Turn on the top and bottom rows of the right segment panel, then submit."""
    frame = env.frame()
    width = len(frame[0])
    segments = [
        blob
        for blob in connected_components(frame, colors=(1, 5), min_area=3)
        if blob.area == 3
        and 1 in blob.size
        and blob.centroid[1] > width / 2
    ]
    if not segments:
        return

    rows = [blob.centroid[0] for blob in segments]
    outer_rows = (min(rows), max(rows))
    for blob in segments:
        if blob.color == 1 and any(
            abs(blob.centroid[0] - row) < 0.5 for row in outer_rows
        ):
            row, col = blob.centroid
            env.step(6, int(round(col)), int(round(row)))

    submit_discs = [
        blob
        for blob in connected_components(frame, colors=(9,), min_area=4)
        if blob.size[0] > 1 and blob.size[1] > 1
    ]
    if submit_discs:
        disc = max(submit_discs, key=lambda blob: blob.area)
        row, col = disc.centroid
        env.step(6, int(round(col)), int(round(row)))
