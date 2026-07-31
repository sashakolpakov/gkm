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
