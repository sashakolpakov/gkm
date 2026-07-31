"""Reproduce solved diagram assignments from levels 2--4."""

import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from perception import connected_components
import players


taint_reason = G._workspace_taint_reason(os.getcwd())
if taint_reason:
    raise SystemExit(taint_reason)


def layout(env):
    blobs = connected_components(env.frame(), min_area=4)
    palette_blobs = [
        b for b in blobs
        if b.bbox[0] >= 48 and b.bbox[2] < 63 and b.area <= 32
    ]
    palette = {b.color: b for b in palette_blobs}
    targets = sorted(
        (b for b in blobs if b.bbox[2] < 16 and b.color in palette and b.area >= 8),
        key=lambda b: b.centroid[1],
    )
    groups = {}
    for b in blobs:
        if 16 < b.centroid[0] < 48 and b.color not in palette and b.area <= 16:
            groups.setdefault(b.color, []).append(b)
    return palette, tuple(b.color for b in targets), groups


def assignment(env):
    palette, targets, groups = layout(env)
    slots = sorted(
        min(
            (g for g in groups.values() if len(g) == len(targets)),
            key=lambda g: sum(b.area for b in g),
        ),
        key=lambda b: b.centroid,
    )
    base = env.levels_completed
    examined = 0

    def search(node, depth, remaining, placed):
        nonlocal examined
        if depth == len(slots):
            examined += 1
            submitted = node.clone()
            submitted.step(5)
            return placed if submitted.levels_completed > base else None
        slot = slots[depth]
        for i, color in enumerate(remaining):
            child = node.clone()
            swatch = palette[color]
            child.step(6, round(swatch.centroid[1]), round(swatch.centroid[0]))
            child.step(6, round(slot.centroid[1]), round(slot.centroid[0]))
            found = search(
                child,
                depth + 1,
                remaining[:i] + remaining[i + 1:],
                placed + (color,),
            )
            if found:
                return found
        return None

    found = search(env.clone(), 0, targets, ())
    print(
        "level",
        base + 1,
        "targets",
        targets,
        "groups",
        {c: [(b.bbox, b.area) for b in g] for c, g in groups.items()},
        "slots",
        [(round(b.centroid[1]), round(b.centroid[0])) for b in slots],
        "assignment",
        found,
        "examined",
        examined,
    )


def probe(env):
    players.play_level_1(env)
    for level in (2, 3, 4):
        assignment(env)
        getattr(players, f"play_level_{level}")(env)


levels, path, err = A.run_program("sb26", probe)
print("done", levels, len(path), err)
