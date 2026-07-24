import sys
from collections import deque

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from legs import MIRRORED_PAIR_ASCENT, MIRRORED_PAIR_MAZE_REUNION
from perception import ACTIONS, action_deltas, color_counts, connected_components


def reach_level_3(env):
    for action in MIRRORED_PAIR_ASCENT + MIRRORED_PAIR_MAZE_REUNION:
        env.step(action)


def compact_blobs(frame):
    return [
        (b.color, b.bbox, b.area)
        for b in connected_components(frame, min_area=4)
        if b.color != 5 or b.area < 3000
    ]


def inspect(env):
    reach_level_3(env)
    frame = np.asarray(env.frame())
    print("level", env.levels_completed + 1, "actions", env.actions)
    print("colors", color_counts(frame))
    print("blobs", compact_blobs(frame))
    print("deltas", {a: {k: v for k, v in d.items() if k != "samples"}
                     for a, d in action_deltas(env, ACTIONS).items()})
    extra = env.clone()
    try:
        extra.step(6)
        print("action6", action_deltas(env, (6,))[6])
    except Exception as exc:
        print("action6_error", type(exc).__name__, str(exc))
    chars = {5: ".", 8: "R", 9: "o", 10: "A", 15: "L"}
    print("map2")
    for r in range(0, 64, 2):
        print("".join(chars[int(frame[r, c])] for c in range(0, 64, 2)))

    queue = deque([(env.clone(), ())])
    seen = {frame.tobytes()}
    interaction = None
    while queue and len(seen) < 10000 and interaction is None:
        node, path = queue.popleft()
        before = np.asarray(node.frame())
        for action in (5, 6):
            child = node.clone()
            child.step(action)
            if child.levels_completed > env.levels_completed or not np.array_equal(before, child.frame()):
                interaction = path + (action,)
                print("interaction", action, "delta",
                      {k: v for k, v in action_deltas(node, (action,))[action].items()
                       if k != "samples"})
                break
        if len(path) >= 80:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            key = np.asarray(child.frame()).tobytes()
            if key not in seen:
                seen.add(key)
                queue.append((child, path + (action,)))
    print("context_search", "seen", len(seen), "path", interaction)
    for prefix in ((1,), (2,), (3,), (4,), (1, 1), (1, 3)):
        context = env.clone()
        for action in prefix:
            context.step(action)
        effects = {}
        for action in (5, 6):
            test = context.clone()
            before = np.asarray(test.frame()).copy()
            test.step(action)
            effects[action] = {
                "delta": action_deltas(context, (action,))[action],
                "colors": color_counts(test.frame()),
                "level": test.levels_completed,
            }
        print("context", prefix, effects)
    for actions in ((1,), (1, 1), (1, 1, 1), (1, 1, 1, 1),
                    (5,), (5, 5), (5, 5, 5), (3, 4, 3, 4)):
        test = env.clone()
        for action in actions:
            test.step(action)
        f = np.asarray(test.frame())
        zeros = list(zip(*np.where(f == 0)))
        avatars = [(b.bbox, b.area) for b in connected_components(f, colors=(10,))]
        print("ticks", actions, "zero", zeros, "avatars", avatars,
              "markers", color_counts(f).get(9, 0))

    def avatar_key(state):
        return np.packbits(np.asarray(state.frame()) == 10).tobytes()

    queue = deque([(env.clone(), ())])
    seen = {avatar_key(env)}
    marker_path = None
    best = None
    targets = ((15, 31), (19, 11), (31, 39))
    while queue and len(seen) < 12000:
        node, path = queue.popleft()
        f = np.asarray(node.frame())
        points = np.argwhere(f == 10)
        score = min(abs(int(r) - tr) + abs(int(c) - tc)
                    for r, c in points for tr, tc in targets)
        if best is None or score < best[0]:
            best = (score, path)
        if color_counts(f).get(9, 0) < 12 or node.levels_completed > env.levels_completed:
            marker_path = path
            break
        if len(path) >= 80:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            key = avatar_key(child)
            if key not in seen:
                seen.add(key)
                queue.append((child, path + (action,)))
    print("avatar_bfs", "seen", len(seen), "best", best, "marker_path", marker_path)
    near = env.clone()
    for action in best[1]:
        near.step(action)
    print("near_objects", compact_blobs(near.frame()))
    for action in (5, 6):
        test = near.clone()
        before = np.asarray(test.frame()).copy()
        test.step(action)
        delta = np.argwhere(before != np.asarray(test.frame()))
        meaningful = [(int(r), int(c), int(before[r, c]), int(np.asarray(test.frame())[r, c]))
                      for r, c in delta if r not in (0, 63)]
        print("near_action", action, "level", test.levels_completed,
              "colors", color_counts(test.frame()), "meaningful", meaningful[:20])
    for label, context in (("start", env), ("near", near)):
        for x, y in ((31, 15), (15, 31), (11, 19), (39, 31),
                     (22, 46), (0, 0)):
            test = context.clone()
            before = np.asarray(test.frame()).copy()
            try:
                test.step(6, x, y)
                delta = np.argwhere(before != np.asarray(test.frame()))
                print("coord", label, (x, y), "level", test.levels_completed,
                      "delta", len(delta), "colors", color_counts(test.frame()))
            except Exception as exc:
                print("coord_error", label, (x, y), type(exc).__name__, str(exc))

    def clean_key(state):
        f = np.asarray(state.frame()).copy()
        f[0, :] = 5
        f[63, :] = 5
        return f.tobytes()

    choices = (1, 2, 3, 4, (6, 31, 15), (6, 11, 19), (6, 39, 31))
    selected = env.clone()
    selected.step(6, 39, 31)
    print("selected_deltas", {
        action: action_deltas(selected, (action,))[action]
        for action in (1, 2, 3, 4, 5, 6)
    })
    for second in ((31, 15), (11, 19), (39, 31), (22, 46)):
        test = selected.clone()
        test.step(6, *second)
        print("second_click", second, "level", test.levels_completed,
              "colors", color_counts(test.frame()))
    queue = deque([(env.clone(), ())])
    seen = {clean_key(env)}
    solution = None
    while queue and len(seen) < 2:
        node, path = queue.popleft()
        if len(path) >= 70:
            continue
        for choice in choices:
            child = node.clone()
            if isinstance(choice, tuple):
                child.step(*choice)
                recorded = (list(choice),)
            else:
                child.step(choice)
                recorded = (choice,)
            new_path = path + recorded
            if child.levels_completed > env.levels_completed:
                solution = new_path
                queue.clear()
                break
            key = clean_key(child)
            if key not in seen:
                seen.add(key)
                queue.append((child, new_path))
    print("mixed_bfs", "seen", len(seen), "solution", solution)


A.run_program("m0r0", inspect)
