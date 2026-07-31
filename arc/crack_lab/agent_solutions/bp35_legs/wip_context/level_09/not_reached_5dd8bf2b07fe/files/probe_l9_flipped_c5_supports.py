"""Find an interaction that keeps the column-five landing below the wall gap."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import ROW_ANCHORS
from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_route_deletions import enter_level_9
from probe_l9_supported_final_alignment import aligned


X_COLUMNS = (3, 9, 15, 21, 27, 33, 39, 45, 51, 57)


def objects(env):
    return tuple(
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(
            env.frame(), colors=(12, 14, 15), min_area=2
        )
        if blob.bbox[0] < 63
    )


def avatar(env):
    return tuple(
        blob.bbox
        for blob in connected_components(env.frame(), colors=(9,), min_area=3)
        if blob.bbox[0] < 63
    )


def lower(env):
    return any(
        blob.bbox[0] < 63
        for blob in connected_components(env.frame(), colors=(14,), min_area=3)
    )


def root_state(root):
    child = aligned(root, 5, 5)
    child.step(*controls(child)[-1])
    return child


def probe(env):
    enter_level_9(env)
    root = root_state(env)
    print(
        "ROOT",
        "avatar",
        avatar(root),
        "grid",
        compact(root)["grid9"],
        "objects",
        objects(root),
        flush=True,
    )
    frame = root.frame()
    targets = tuple(
        (x, y, int(frame[y][x]))
        for y in ROW_ANCHORS
        for x in X_COLUMNS
        if int(frame[y][x]) in (12, 14, 15)
    )
    print("TARGETS", targets, flush=True)
    for x, y, color in targets:
        child = root.clone()
        before = objects(child)
        child.step(6, x, y)
        after = objects(child)
        changed = before != after
        if child.terminal():
            print("CLICK_TERMINAL", (x, y, color), flush=True)
            continue
        child.step(3)
        if not child.terminal():
            child.step(3)
        print(
            "TEST",
            (x, y, color),
            "changed",
            changed,
            "terminal",
            bool(child.terminal()),
            "lower",
            lower(child),
            "avatar",
            avatar(child),
            "grid",
            compact(child)["grid9"],
            flush=True,
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
