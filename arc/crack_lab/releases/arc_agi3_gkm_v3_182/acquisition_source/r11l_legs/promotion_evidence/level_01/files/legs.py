"""Reusable skills derived from public frames and clone experiments."""
from perception import arr, connected_components, replay, safe_step


def center_link_on_ring(env):
    """Place two swappable endpoints around an outlined midpoint target.

    Black is the movable endpoint; clicking green swaps the endpoints.
    The white marker tracks their integer midpoint. Verify a symmetric
    placement on a clone before committing its three coordinate clicks.
    """
    frame = arr(env.frame())
    endpoints = [b for b in connected_components(frame, colors=[0, 3])
                 if b.area == 12 and b.size == (5, 5)]
    if len(endpoints) != 2:
        return
    green = next(b for b in endpoints if b.color == 3)
    swap = (6, int(green.centroid[1]), int(green.centroid[0]))
    markers = connected_components(frame, colors=[15])
    # The target is an outline of isolated white pixels. Exclude endpoint
    # centers and the solid white midpoint marker.
    outline = [b for b in markers if b.area == 1 and not any(
        e.bbox[0] <= b.centroid[0] <= e.bbox[2]
        and e.bbox[1] <= b.centroid[1] <= e.bbox[3]
        for e in endpoints)]
    if len(outline) < 4:
        return
    tx = round(sum(b.centroid[1] for b in outline) / len(outline))
    ty = round(sum(b.centroid[0] for b in outline) / len(outline))
    base = env.levels_completed
    for dx, dy in [(8, 0), (0, 8), (-8, 0), (0, -8)]:
        a, b = (tx + dx, ty + dy), (tx - dx, ty - dy)
        if not all(2 <= x <= 61 and 2 <= y <= 61
                   and frame[y, x] != 2 for x, y in (a, b)):
            continue
        # Stage an endpoint near the target, swap to hold it fixed, then
        # reduce midpoint distance to zero with the symmetric endpoint.
        path = [(6, *a), swap, (6, *b)]
        trial = replay(env, path)
        if trial.levels_completed > base:
            for action in path:
                if env.terminal() or env.levels_completed > base:
                    break
                safe_step(env, action)
            return
