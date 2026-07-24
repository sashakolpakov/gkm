"""Compact observational probes for the current g50t level-6 frontier."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np
from collections import deque

import perception as p
import legs
import solve


def compact_objects(frame):
    return [
        (o["color"], o["bbox"], o["area"])
        for o in p.object_candidates(frame, min_area=4)
        if o["color"] != 1
    ]


def translated_components(before, after):
    out = []
    for color in sorted(set(p.color_counts(before)) | set(p.color_counts(after))):
        old = {(b.bbox, b.area) for b in p.connected_components(before, (color,), 4)}
        new = {(b.bbox, b.area) for b in p.connected_components(after, (color,), 4)}
        if old != new:
            out.append((color, sorted(old - new), sorted(new - old)))
    return out


def actor_positions(frame):
    f = np.asarray(frame)
    avatar = legs._avatar_pos(f)
    hr, hc = np.where(f == 14)
    helper = () if not len(hr) else ((int(hr.min()), int(hc.min())),)
    return ((9, () if avatar is None else (avatar,)), (14, helper))


def tile_map(frame):
    f = np.asarray(frame)
    rows = []
    for r in range(8, 62, 6):
        row = []
        for c in range(2, 62, 6):
            vals, counts = np.unique(f[r:r + 5, c:c + 5], return_counts=True)
            order = np.argsort(-counts)
            sig = "".join("0123456789ABCDEF"[int(vals[i])] for i in order
                          if int(vals[i]) != 0)
            row.append(sig or ".")
        rows.append(" ".join(f"{x:2}" for x in row))
    return rows


def trace(base, actions):
    node = base.clone()
    states = [actor_positions(node.frame())]
    for action in actions:
        node.step(action)
        states.append(actor_positions(node.frame()))
    return states, int(node.levels_completed)


def movement_bfs(base, max_states=4000, max_depth=35, scan_use=False):
    q = deque([(base.clone(), [])])
    seen = {actor_positions(base.frame())}
    state_paths = {actor_positions(base.frame()): []}
    by_avatar = {}
    use_effects = []
    while q and len(seen) < max_states:
        node, path = q.popleft()
        actors = actor_positions(node.frame())
        avatar = dict(actors).get(9, ())
        if avatar and avatar[0] not in by_avatar:
            by_avatar[avatar[0]] = path
        if len(path) >= max_depth:
            continue
        if scan_use:
            use = node.clone()
            before = np.asarray(node.frame())
            use.step(5)
            delta = p.frame_delta(before, use.frame())
            reset_delta = p.frame_delta(np.asarray(base.frame())[7:-1],
                                        np.asarray(use.frame())[7:-1])
            if reset_delta["count"] or int(use.levels_completed) > int(base.levels_completed):
                trans = {}
                initial = np.asarray(base.frame())[7:-1]
                committed = np.asarray(use.frame())[7:-1]
                for old, new in zip(initial[initial != committed], committed[initial != committed]):
                    trans[(int(old), int(new))] = trans.get((int(old), int(new)), 0) + 1
                use_effects.append((actors, len(path), reset_delta["count"],
                                    reset_delta["bbox"], sorted(trans.items()),
                                    int(use.levels_completed), path + [5]))
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            if int(child.levels_completed) > int(base.levels_completed):
                return path + [action], state_paths, by_avatar, use_effects
            key = actor_positions(child.frame())
            if key in seen:
                continue
            seen.add(key)
            state_paths[key] = path + [action]
            q.append((child, path + [action]))
    return None, state_paths, by_avatar, use_effects


def probe(env):
    solve.solve(env)
    if len(sys.argv) > 1 and sys.argv[1] == "policy":
        node = env.clone()
        prefix = []
        for stage in range(40):
            win, states, positions, _ = movement_bfs(node)
            print("FAST_STAGE", stage, len(states), len(positions), win,
                  flush=True)
            if win is not None:
                print("FAST_PLAN", prefix + win, flush=True)
                return
            vanished = [(len(path), path) for key, path in states.items()
                        if not dict(key).get(14, ())]
            if not vanished:
                return
            _, path = min(vanished)
            prefix += path + [5]
            node = p.replay(node, path + [5])
        return
    base = np.asarray(env.frame()).copy()
    print("LEVEL", int(env.levels_completed), "ACTIONS", tuple(env.actions))
    print("COLORS", p.color_counts(base))
    print("OBJECTS", compact_objects(base))
    print("TILES")
    for row in tile_map(base):
        print(row)
    for action in env.actions:
        child = env.clone()
        child.step(action)
        delta = p.frame_delta(base, child.frame())
        print(
            "ACTION", action,
            "REWARD", int(child.levels_completed),
            "DELTA", (delta["count"], delta["bbox"]),
            "COMPS", translated_components(base, child.frame()),
        )
    for actions in ((2, 2, 2, 2, 2), (3, 3, 3, 3, 3),
                    (2, 3, 2, 3, 2), (3, 2, 4, 1, 5)):
        states, reward = trace(env, actions)
        print("TRACE", actions, "REWARD", reward, "ACTORS", states)
    win, states, positions, use_effects = movement_bfs(env, scan_use=True)
    print("MOVE_BFS", "WIN", win, "PAIR_STATES", len(states),
          "AVATAR_POSITIONS", len(positions),
          "POS", sorted(positions), "USE_EFFECTS", use_effects[:20])
    print("JOINT_PATHS", sorted((key, path) for key, path in states.items()))

    node = env.clone()
    prefix = []
    for stage in range(8):
        win, stage_states, stage_pos, _ = movement_bfs(node)
        helpers = sorted({dict(key).get(14, ()) for key in stage_states})
        print("STAGE", stage, "STATES", len(stage_states),
              "AVATARS", len(stage_pos), "HELPERS", helpers, "WIN", win)
        if win is not None:
            print("STAGE_PLAN", prefix + win)
            break
        vanished = [(len(path), path, key) for key, path in stage_states.items()
                    if not dict(key).get(14, ())]
        if not vanished:
            break
        _, path, key = min(vanished)
        print("COMMIT_VANISHED", key, path + [5])
        prefix += path + [5]
        node = p.replay(node, path + [5])

    for target in ((),):
        node = env.clone()
        prefix = []
        summary = []
        for stage in range(40):
            win, stage_states, _, _ = movement_bfs(node)
            summary.append(len(stage_states))
            if win is not None:
                prefix += win
                break
            choices = [(len(path), path) for key, path in stage_states.items()
                       if dict(key).get(14, ()) == target]
            if not choices:
                break
            _, path = min(choices)
            prefix += path + [5]
            node = p.replay(node, path + [5])
        print("POLICY", target, "COUNTS", summary, "REWARD",
              int(node.levels_completed), "PLAN", prefix if win is not None else None)


arena.run_program("g50t", probe)
