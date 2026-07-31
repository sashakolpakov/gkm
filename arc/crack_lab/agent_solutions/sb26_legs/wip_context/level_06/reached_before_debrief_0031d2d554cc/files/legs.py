# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.

from collections import Counter
from itertools import permutations

from perception import connected_components


def copy_visible_color_code(env, click_action=6, submit_action=5):
    """Copy an ordered top color code into central slots using a bottom palette."""
    frame = env.frame()
    height, width = frame.shape[:2]
    blobs = connected_components(frame, min_area=4)

    palette_blobs = [
        blob for blob in blobs
        if blob.bbox[0] >= 3 * height // 4
        and blob.bbox[2] < height - 1
        and blob.area <= 32
    ]
    palette = {blob.color: blob for blob in palette_blobs}
    targets = sorted(
        (
            blob for blob in blobs
            if blob.bbox[2] < height // 4
            and blob.color in palette
            and blob.area >= 8
        ),
        key=lambda blob: blob.centroid[1],
    )
    slot_groups = {}
    for blob in blobs:
        if (
            height // 4 < blob.centroid[0] < 3 * height // 4
            and blob.area <= 16
            and blob.color not in palette
        ):
            slot_groups.setdefault(blob.color, []).append(blob)
    aligned_groups = [
        group for group in slot_groups.values()
        if len(group) == len(targets)
        and max(blob.centroid[0] for blob in group)
        - min(blob.centroid[0] for blob in group) <= height // 16
    ]
    slots = sorted(
        min(
            aligned_groups,
            key=lambda group: max(blob.centroid[0] for blob in group)
            - min(blob.centroid[0] for blob in group),
        )
        if aligned_groups else [],
        key=lambda blob: blob.centroid[1],
    )

    if not targets or len(targets) != len(slots):
        raise RuntimeError(
            f"color-code layout not recognized: targets={len(targets)} slots={len(slots)}"
        )

    for target, slot in zip(targets, slots):
        swatch = palette[target.color]
        env.step(
            click_action,
            int(round(swatch.centroid[1])),
            int(round(swatch.centroid[0])),
        )
        if env.terminal():
            return
        env.step(
            click_action,
            int(round(slot.centroid[1])),
            int(round(slot.centroid[0])),
        )
        if env.terminal():
            return

    env.step(submit_action)


def copy_visible_color_code_into_diagram(
    env, click_action=6, submit_action=5, max_assignments=6000
):
    """Place a visible color code into diagram cells, discovering their order."""
    frame = env.frame()
    height, width = frame.shape[:2]
    blobs = connected_components(frame, min_area=4)

    palette_blobs = [
        blob for blob in blobs
        if blob.bbox[0] >= 3 * height // 4
        and blob.bbox[2] < height - 1
        and blob.area <= 32
    ]
    palette = {blob.color: blob for blob in palette_blobs}
    targets = sorted(
        (
            blob for blob in blobs
            if blob.bbox[2] < height // 4
            and blob.color in palette
            and blob.area >= 8
        ),
        key=lambda blob: blob.centroid[1],
    )
    target_colors = tuple(blob.color for blob in targets)
    slot_groups = {}
    for blob in blobs:
        if (
            height // 4 < blob.centroid[0] < 3 * height // 4
            and blob.color not in palette
            and blob.area <= 16
        ):
            slot_groups.setdefault(blob.color, []).append(blob)
    matching_groups = [
        group for group in slot_groups.values()
        if len(group) == len(target_colors)
    ]
    slots = sorted(
        min(matching_groups, key=lambda group: sum(blob.area for blob in group))
        if matching_groups else [],
        key=lambda blob: blob.centroid,
    )

    if (
        not target_colors
        or len(target_colors) != len(slots)
        or len(set(target_colors)) != len(target_colors)
    ):
        raise RuntimeError(
            "color-diagram layout not recognized: "
            f"targets={len(target_colors)} slots={len(slots)}"
        )

    base_level = env.levels_completed
    examined = 0

    def find_assignment(node, depth, remaining, assignment):
        nonlocal examined
        if depth == len(slots):
            examined += 1
            if examined > max_assignments:
                return None
            submitted = node.clone()
            submitted.step(submit_action)
            if submitted.levels_completed > base_level:
                return assignment
            return None

        slot = slots[depth]
        for index, color in enumerate(remaining):
            child = node.clone()
            swatch = palette[color]
            child.step(
                click_action,
                int(round(swatch.centroid[1])),
                int(round(swatch.centroid[0])),
            )
            child.step(
                click_action,
                int(round(slot.centroid[1])),
                int(round(slot.centroid[0])),
            )
            result = find_assignment(
                child,
                depth + 1,
                remaining[:index] + remaining[index + 1:],
                assignment + (color,),
            )
            if result is not None:
                return result
            if examined >= max_assignments:
                return None
        return None

    assignment = find_assignment(
        env.clone(), 0, target_colors, ()
    )
    if assignment is None:
        raise RuntimeError(
            f"no color-diagram assignment found in {examined} candidates"
        )

    for slot, color in zip(slots, assignment):
        swatch = palette[color]
        env.step(
            click_action,
            int(round(swatch.centroid[1])),
            int(round(swatch.centroid[0])),
        )
        if env.terminal():
            return
        env.step(
            click_action,
            int(round(slot.centroid[1])),
            int(round(slot.centroid[0])),
        )
        if env.terminal():
            return
    env.step(submit_action)


def copy_repeated_code_into_nested_diagram(
    env, click_action=6, submit_action=5
):
    """Factor a repeated code block into parent call cells and a child row."""
    frame = env.frame()
    height, width = frame.shape[:2]
    background = int(frame[height // 2, 0])
    blobs = connected_components(frame, min_area=4)

    palette_tiles = sorted(
        (
            blob for blob in blobs
            if blob.bbox[0] >= 7 * height // 8
            and blob.bbox[2] < height - 1
            and blob.color != background
            and 2 <= blob.size[0] <= 5
            and 2 <= blob.size[1] <= 5
            and blob.area <= 24
        ),
        key=lambda blob: blob.centroid[1],
    )
    palette_colors = {blob.color for blob in palette_tiles}
    target = tuple(
        blob.color for blob in sorted(
            (
                blob for blob in blobs
                if blob.bbox[2] < height // 8
                and blob.color in palette_colors
                and blob.area >= 8
            ),
            key=lambda blob: blob.centroid[1],
        )
    )

    row_groups = {}
    for blob in blobs:
        if (
            height // 4 < blob.centroid[0] < 3 * height // 4
            and blob.area <= 8
            and blob.size[0] <= 3
            and blob.size[1] <= 3
        ):
            key = (blob.color, int(round(blob.centroid[0])))
            row_groups.setdefault(key, []).append(blob)
    slot_rows = sorted(
        (
            sorted(group, key=lambda blob: blob.centroid[1])
            for group in row_groups.values()
            if len(group) >= 3
        ),
        key=lambda group: group[0].centroid[0],
    )

    if not target or len(slot_rows) != 2:
        raise RuntimeError(
            "nested color-diagram layout not recognized: "
            f"targets={len(target)} rows={[len(row) for row in slot_rows]}"
        )

    parent_slots, child_slots = slot_rows
    palette_counts = Counter(blob.color for blob in palette_tiles)
    arrangement = None
    for call_color, repetitions in sorted(palette_counts.items()):
        if repetitions < 2:
            continue
        repeated_length = repetitions * len(child_slots)
        for start in range(len(target) - repeated_length + 1):
            child_colors = target[start:start + len(child_slots)]
            if (
                target[start:start + repeated_length]
                != child_colors * repetitions
            ):
                continue
            parent_colors = (
                target[:start]
                + (call_color,) * repetitions
                + target[start + repeated_length:]
            )
            if (
                len(parent_colors) == len(parent_slots)
                and Counter(parent_colors) + Counter(child_colors)
                == palette_counts
            ):
                arrangement = (
                    tuple(parent_slots) + tuple(child_slots),
                    parent_colors + child_colors,
                )
                break
        if arrangement is not None:
            break

    if arrangement is None:
        raise RuntimeError("visible code has no palette-preserving nested factor")

    slots, colors = arrangement
    tile_positions = {}
    for tile in palette_tiles:
        tile_positions.setdefault(tile.color, []).append(
            (
                int(round(tile.centroid[1])),
                int(round(tile.centroid[0])),
            )
        )

    choices = []
    for slot, color in zip(slots, colors):
        choices.append(
            (
                tile_positions[color].pop(0),
                (
                    int(round(slot.centroid[1])),
                    int(round(slot.centroid[0])),
                ),
            )
        )

    def place_and_submit(node):
        for tile, slot in choices:
            node.step(click_action, *tile)
            node.step(click_action, *slot)
        node.step(submit_action)

    base_level = env.levels_completed
    verified = env.clone()
    place_and_submit(verified)
    if verified.levels_completed <= base_level:
        raise RuntimeError("nested color-diagram factor failed clone verification")
    place_and_submit(env)


def factor_visible_code_into_parallel_diagrams(
    env, click_action=6, submit_action=5
):
    """Split a preorder color code across a parent and several child diagrams."""
    frame = env.frame()
    height, width = frame.shape[:2]
    background = int(frame[height // 2, 0])
    blobs = connected_components(frame, min_area=2)

    palette_tiles = sorted(
        (
            blob for blob in blobs
            if blob.bbox[0] >= 7 * height // 8
            and blob.bbox[2] < height - 1
            and blob.color != background
            and 2 <= blob.size[0] <= 5
            and 2 <= blob.size[1] <= 5
            and blob.area <= 24
        ),
        key=lambda blob: blob.centroid[1],
    )
    palette_colors = {blob.color for blob in palette_tiles}
    target = tuple(
        blob.color for blob in sorted(
            (
                blob for blob in blobs
                if blob.bbox[2] < height // 8
                and blob.color in palette_colors
                and blob.area >= 8
            ),
            key=lambda blob: blob.centroid[1],
        )
    )

    small_slots = [
        blob for blob in blobs
        if height // 4 < blob.centroid[0] < 3 * height // 4
        and blob.area <= 8
        and blob.size[0] <= 3
        and blob.size[1] <= 3
    ]
    diagrams = []
    for border in blobs:
        border_height, border_width = border.size
        if not (
            height // 4 < border.centroid[0] < 3 * height // 4
            and border.color != background
            and 6 <= border_height <= height // 4
            and 12 <= border_width <= width // 2
            and border.area >= 2 * (border_height + border_width) - 8
        ):
            continue
        r0, c0, r1, c1 = border.bbox
        slots = sorted(
            (
                slot for slot in small_slots
                if r0 < slot.centroid[0] < r1
                and c0 < slot.centroid[1] < c1
            ),
            key=lambda slot: slot.centroid[1],
        )
        if slots:
            diagrams.append((border.color, tuple(slots)))

    roots = [
        diagram for diagram in diagrams
        if diagram[0] not in palette_colors
    ]
    children = [
        diagram for diagram in diagrams
        if diagram[0] in palette_colors
    ]
    if (
        not target
        or len(roots) != 1
        or len(roots[0][1]) != len(children)
        or not children
    ):
        raise RuntimeError(
            "parallel diagram layout not recognized: "
            f"targets={len(target)} roots={len(roots)} children={len(children)}"
        )

    root_slots = roots[0][1]
    arrangement = None
    for ordered_children in permutations(children):
        offset = 0
        child_colors = []
        valid = True
        for call_color, child_slots in ordered_children:
            end = offset + 1 + len(child_slots)
            chunk = target[offset:end]
            if len(chunk) != 1 + len(child_slots) or chunk[0] != call_color:
                valid = False
                break
            child_colors.append(tuple(chunk[1:]))
            offset = end
        if not valid or offset != len(target):
            continue
        colors = (
            tuple(call_color for call_color, _ in ordered_children)
            + tuple(color for group in child_colors for color in group)
        )
        if Counter(colors) == Counter(blob.color for blob in palette_tiles):
            slots = (
                tuple(root_slots)
                + tuple(
                    slot
                    for (_, child_slots) in ordered_children
                    for slot in child_slots
                )
            )
            arrangement = slots, colors
            break

    if arrangement is None:
        raise RuntimeError("visible code has no palette-preserving parallel factor")

    tile_positions = {}
    for tile in palette_tiles:
        tile_positions.setdefault(tile.color, []).append(
            (
                int(round(tile.centroid[1])),
                int(round(tile.centroid[0])),
            )
        )
    choices = []
    for slot, color in zip(*arrangement):
        choices.append(
            (
                tile_positions[color].pop(0),
                (
                    int(round(slot.centroid[1])),
                    int(round(slot.centroid[0])),
                ),
            )
        )

    def place_and_submit(node):
        for tile, slot in choices:
            node.step(click_action, *tile)
            node.step(click_action, *slot)
        node.step(submit_action)

    base_level = env.levels_completed
    verified = env.clone()
    place_and_submit(verified)
    if verified.levels_completed <= base_level:
        raise RuntimeError("parallel diagram factor failed clone verification")
    place_and_submit(env)
