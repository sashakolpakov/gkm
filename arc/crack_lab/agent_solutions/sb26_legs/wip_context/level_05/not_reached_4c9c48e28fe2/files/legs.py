# Shared leg library: small, named, reusable skills.
# Players import from here; add a NEW leg only when no existing leg fits.

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


def arrange_visible_palette_multiset_into_diagram(
    env, click_action=6, submit_action=5, max_assignments=10080
):
    """Arrange all visible palette tiles into equally many diagram cells."""
    frame = env.frame()
    height, width = frame.shape[:2]
    blobs = connected_components(frame, min_area=4)
    background = max(blobs, key=lambda blob: blob.area).color

    palette_tiles = sorted(
        (
            blob for blob in blobs
            if blob.bbox[0] >= 3 * height // 4
            and blob.bbox[2] < height - 1
            and blob.area <= 32
            and blob.color != background
        ),
        key=lambda blob: blob.centroid[1],
    )

    slot_groups = {}
    for blob in blobs:
        if (
            height // 4 < blob.centroid[0] < 3 * height // 4
            and blob.area <= 16
            and blob.color != background
        ):
            slot_groups.setdefault(blob.color, []).append(blob)
    matching_groups = [
        group for group in slot_groups.values()
        if len(group) == len(palette_tiles)
    ]
    slots = sorted(
        min(matching_groups, key=lambda group: sum(blob.area for blob in group))
        if matching_groups else [],
        key=lambda blob: blob.centroid,
    )

    if not palette_tiles or len(palette_tiles) != len(slots):
        raise RuntimeError(
            "palette-diagram layout not recognized: "
            f"palette={len(palette_tiles)} slots={len(slots)}"
        )

    # Repeated palette entries can share a colour and shape. Observe what each
    # tile paints, then search unique multiset arrangements rather than all
    # permutations of indistinguishable entries.
    sample_slot = slots[0]
    r0, c0, r1, c1 = sample_slot.bbox
    tile_types = []
    type_by_signature = {}
    for swatch in palette_tiles:
        probe = env.clone()
        probe.step(
            click_action,
            int(round(swatch.centroid[1])),
            int(round(swatch.centroid[0])),
        )
        probe.step(
            click_action,
            int(round(sample_slot.centroid[1])),
            int(round(sample_slot.centroid[0])),
        )
        signature = probe.frame()[
            max(0, r0 - 1):min(height, r1 + 2),
            max(0, c0 - 1):min(width, c1 + 2),
        ].tobytes()
        type_index = type_by_signature.get(signature)
        if type_index is None:
            type_by_signature[signature] = len(tile_types)
            tile_types.append([swatch, 1])
        else:
            tile_types[type_index][1] += 1

    base_level = env.levels_completed
    examined = 0

    def find_assignment(node, depth, remaining, assignment):
        nonlocal examined
        if depth == len(slots):
            if examined >= max_assignments:
                return None
            examined += 1
            submitted = node.clone()
            submitted.step(submit_action)
            if submitted.levels_completed > base_level:
                return assignment
            return None

        slot = slots[depth]
        for type_index, (swatch, _) in enumerate(tile_types):
            if remaining[type_index] == 0:
                continue
            child = node.clone()
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
            next_remaining = list(remaining)
            next_remaining[type_index] -= 1
            result = find_assignment(
                child,
                depth + 1,
                tuple(next_remaining),
                assignment + (type_index,),
            )
            if result is not None:
                return result
            if examined >= max_assignments:
                return None
        return None

    assignment = find_assignment(
        env.clone(),
        0,
        tuple(count for _, count in tile_types),
        (),
    )
    if assignment is None:
        raise RuntimeError(
            f"no palette-diagram assignment found in {examined} candidates"
        )

    for slot, type_index in zip(slots, assignment):
        swatch = tile_types[type_index][0]
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
