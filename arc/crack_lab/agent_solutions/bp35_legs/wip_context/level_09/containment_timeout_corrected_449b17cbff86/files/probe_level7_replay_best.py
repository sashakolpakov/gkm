import heapq
import itertools
import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
import gkm_legs as G

from legs import _cell_shape, band_shift, click_action, moves_used
from perception import arr
from probe_level7_no_control import PREFIX, advance, avatar_cell, controls


if G._workspace_taint_reason(os.getcwd()):
    raise SystemExit("tainted workspace")


def action_choices(frame):
    actions = [(3,), (4,)]
    actions += [(6, 3, y) for y in controls(frame)]
    avatar = avatar_cell(frame)
    if avatar is not None:
        ai, aj = avatar
        actions += [
            click_action(i, j)
            for i in range(max(0, ai - 2), min(10, ai + 3))
            for j in range(max(0, aj - 1), min(8, aj + 2))
            if _cell_shape(frame, i, j)[0] in (12, 14)
        ]
    return list(dict.fromkeys(actions))


def search(root, max_expansions=400, max_depth=24):
    counter = itertools.count()

    def reconstruct(path):
        node = root.clone()
        for action in path:
            if node.terminal():
                break
            node.step(*action)
        return node

    def key(node, height):
        frame = arr(node.frame())
        return height, frame[:63].tobytes(), moves_used(frame) % 2

    def score(node, height):
        frame = node.frame()
        avatar = avatar_cell(frame)
        central = (
            0 if avatar is None else min(avatar[1], 7 - avatar[1], 3)
        )
        return height * 10 + len(controls(frame)) * 4 + central

    start = reconstruct(())
    queue = [(-score(start, 0), 0, next(counter), 0, ())]
    seen = {key(start, 0)}
    best = (0, len(controls(start.frame())), ())
    expanded = 0
    while queue and expanded < max_expansions:
        _, _, _, height, path = heapq.heappop(queue)
        node = reconstruct(path)
        expanded += 1
        if expanded % 50 == 0:
            print(
                "EXPANDED", expanded, "QUEUE", len(queue),
                "BEST", best[:2], flush=True,
            )
        if len(path) >= max_depth or node.terminal():
            continue
        before = arr(node.frame()).copy()
        for action in action_choices(before):
            child_path = (*path, action)
            child = reconstruct(child_path)
            if child.levels_completed > 6:
                return list(child_path), expanded, best
            if child.terminal() or avatar_cell(child.frame()) is None:
                continue
            child_height = height + band_shift(before, child.frame())
            child_key = key(child, child_height)
            if child_key in seen:
                continue
            seen.add(child_key)
            progress = (
                child_height,
                len(controls(child.frame())),
                child_path,
            )
            if progress[:2] > best[:2]:
                best = progress
                print(
                    "PROGRESS", expanded, child_height,
                    len(controls(child.frame())), len(child_path),
                    child_path, flush=True,
                )
            heapq.heappush(
                queue,
                (
                    -score(child, child_height),
                    len(child_path),
                    next(counter),
                    child_height,
                    child_path,
                ),
            )
    return [], expanded, best


def probe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    root = env
    prefix_height = advance(root, PREFIX)
    print(
        "ROOT", prefix_height, avatar_cell(root.frame()),
        controls(root.frame()), flush=True,
    )
    route, expanded, best = search(root)
    print("SEARCH", expanded, len(route), route)
    print("BEST", best)
    if route:
        verified = root.clone()
        advance(verified, route)
        print(
            "VERIFY", verified.levels_completed, verified.terminal(),
            avatar_cell(verified.frame()), controls(verified.frame()),
        )
        print("WIN", [*PREFIX, *route], flush=True)


levels, path, err = A.run_program("bp35", probe)
print("RESULT", levels, len(path), err)
