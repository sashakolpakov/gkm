"""Compare gravity-flip timing before the far-right shaft loses all margin."""

import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import connected_components
from probe_l9_control_row import compact, controls
from probe_l9_stage_under_wall import enter_wall_outside


def avatars(env):
    return [
        blob
        for blob in connected_components(env.frame(), colors=(9, 11), min_area=3)
        if blob.bbox[0] < 63
    ]


def above_catch(env):
    avs = avatars(env)
    if not avs:
        return None
    avatar = avs[0]
    ay0, ax0, _, ax1 = avatar.bbox
    candidates = []
    for blob in connected_components(env.frame(), colors=(15,), min_area=3):
        y0, x0, y1, x1 = blob.bbox
        if (
            blob.area == 21
            and y1 < ay0
            and x0 <= ax1
            and x1 >= ax0
        ):
            candidates.append(blob)
    if not candidates:
        return None
    blob = max(candidates, key=lambda item: item.bbox[2])
    return 6, round(blob.centroid[1]), round(blob.centroid[0])


def goals(env):
    return [
        (blob.bbox, blob.area)
        for blob in connected_components(env.frame(), colors=(7,), min_area=3)
        if blob.bbox[0] < 63
    ]


def probe(env):
    base = env.clone()
    for depth in range(5, 9):
        child = base.clone()
        enter_wall_outside(child)
        for _ in range(depth):
            child.step(6, 57, 35)
        visible = controls(child)
        before = compact(child)
        child.step(*visible[0])
        after_flip = compact(child)
        safe = 0
        route = []
        for _ in range(10):
            target = above_catch(child)
            if not target or child.terminal():
                break
            route.append(target)
            child.step(*target)
            if child.terminal():
                break
            safe += 1
            if int(child.levels_completed) >= 9:
                break
        print(
            "TIMING",
            depth,
            "control",
            visible[0],
            "before",
            before,
            "after_flip",
            after_flip,
            "route",
            route,
            "safe",
            safe,
            "terminal",
            bool(child.terminal()),
            "goals",
            goals(child),
            "final",
            compact(child),
        )


if __name__ == "__main__":
    arena.run_program("bp35", probe)
