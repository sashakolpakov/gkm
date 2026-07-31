# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.

from perception import arr, connected_components


UP, DOWN, RETRACT, EXTEND = 1, 2, 3, 4


def assemble_telescoping_chain(env):
    """Build the colored chain shown in the lower guide.

    The carriage moves vertically, while its sticky arm extends horizontally.
    A contacted block joins the end of the chain.  Retraction stops as soon as
    the connected prefix is parked beside the carriage, preserving the tiny
    connectors between blocks.
    """
    initial = arr(env.frame())
    height, width = initial.shape[:2]
    blobs = connected_components(initial, min_area=1)

    separators = [
        b.bbox[0] for b in blobs
        if b.size == (1, width) and b.area == width
    ]
    if not separators:
        return
    separator = max(r for r in separators if r < height - 4)

    guide = [
        b for b in blobs
        if b.bbox[0] > separator and b.size == (4, 4) and b.area == 16
    ]
    guide.sort(key=lambda b: b.bbox[1])
    order = [b.color for b in guide]
    if not order:
        return

    guide_area = {b.color: b.area for b in guide}
    guide_cols = [b.bbox[1] for b in guide]
    spacings = [b - a for a, b in zip(guide_cols, guide_cols[1:]) if b > a]
    step = min(spacings) if spacings else 6

    top_blocks = {
        b.color: b.bbox
        for b in blobs
        if b.color in order and b.bbox[2] < separator
        and b.size == (4, 4) and b.area == 16
    }
    if any(color not in top_blocks for color in order):
        return
    target_rows = {color: top_blocks[color][0] for color in order}

    def live_blobs():
        return connected_components(env.frame(), min_area=1)

    def head_box():
        candidates = [
            b for b in live_blobs()
            if b.color == 0 and b.bbox[2] < separator and b.size == (4, 4)
        ]
        return candidates[0].bbox if candidates else None

    def positions(colors):
        wanted = set(colors)
        return {
            b.color: b.bbox
            for b in live_blobs()
            if b.color in wanted and b.bbox[2] < separator
            and b.size == (4, 4) and b.area == 16
        }

    def guide_is_marked(color):
        candidates = [
            b for b in live_blobs()
            if b.color == color and b.bbox[0] > separator and b.size == (4, 4)
        ]
        return bool(candidates and candidates[0].area < guide_area[color])

    base_level = env.levels_completed
    collected = []
    for color in order:
        head = head_box()
        if head is None:
            return
        vertical = target_rows[color] - head[0]
        action = DOWN if vertical > 0 else UP
        for _ in range(abs(vertical) // step):
            env.step(action)

        # Extend until the guide confirms that the expected color joined.
        attached = False
        for _ in range(width // step):
            env.step(EXTEND)
            if env.levels_completed > base_level:
                return
            if guide_is_marked(color):
                attached = True
                break
        if not attached:
            return
        collected.append(color)

        # Park the connected prefix beside the carriage.  Stopping on this
        # observable layout avoids an extra retract that would sever its tail.
        for _ in range(width // step):
            head = head_box()
            placed = positions(collected)
            if head is None or len(placed) != len(collected):
                return
            if all(
                placed[item][0] == head[0]
                and placed[item][1] == head[1] + step * (index + 1)
                for index, item in enumerate(collected)
            ):
                break
            env.step(RETRACT)
