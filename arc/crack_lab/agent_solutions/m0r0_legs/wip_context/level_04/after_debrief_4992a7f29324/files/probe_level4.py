import sys
from collections import deque

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

import perception as p
import solve
import legs


def coarse(frame):
    f = p.arr(frame)
    chars = {1: "i", 5: ".", 8: "#", 9: "G", 10: "A", 11: "l", 15: "r"}
    return ["".join(chars.get(int(f[r, c]), "?") for c in range(1, 64, 5))
            for r in range(1, 64, 5)]


def compact_objects(frame):
    return [
        (o["color"], o["bbox"], o["area"])
        for o in p.object_candidates(frame)
        if o["color"] != 5
    ]


def blobs(frame, colors=(8, 9, 10)):
    return [(b.color, b.bbox, b.area) for b in p.connected_components(frame, colors)]


def agents(frame):
    return [b.bbox for b in p.connected_components(frame, (10,), min_area=20)]


def closest_pair(env):
    q = deque([(env.clone(), [])])
    seen = set()
    best = None
    while q and len(seen) < 20000:
        node, path = q.popleft()
        boxes = agents(node.frame())
        key = tuple(boxes)
        if key in seen:
            continue
        seen.add(key)
        if len(boxes) == 1:
            return len(seen), 0, path, boxes
        if len(boxes) == 2:
            centers = [((a + c) // 2, (b + d) // 2) for a, b, c, d in boxes]
            score = abs(centers[0][0] - centers[1][0]) + abs(centers[0][1] - centers[1][1])
            if best is None or score < best[0]:
                best = (score, path, boxes)
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            q.append((child, path + [action]))
    return len(seen), *best


def use_contexts(env):
    q = deque([(env.clone(), [])])
    seen = set()
    effects = []

    def key(node):
        return (
            int(node.levels_completed),
            tuple((b.color, b.bbox, b.area)
                  for b in p.connected_components(node.frame(), (1, 9, 10))),
        )

    while q:
        node, path = q.popleft()
        state_key = key(node)
        if state_key in seen:
            continue
        seen.add(state_key)
        used = node.clone()
        used.step(5)
        for action in (1, 2, 3, 4, 5):
            direct = node.clone()
            direct.step(action)
            after_use = used.clone()
            after_use.step(action)
            if key(direct) != key(after_use):
                effects.append((path, action, key(direct), key(after_use)))
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            q.append((child, path + [action]))
    return len(seen), effects[:3]


def mixed_bfs(env):
    selected = env.clone()
    selected.step(6, 31, 31)
    q = deque([(selected, [])])
    seen_small = set()
    while q and len(seen_small) < 300:
        node, small_path = q.popleft()
        small_key = tuple(revealed_small(node))
        if small_key in seen_small:
            continue
        seen_small.add(small_key)

        pair_state = node.clone()
        pair_state.step(6, 46, 26)
        pair_path = p.bounded_bfs(
            pair_state,
            p.level_goal(env.levels_completed),
            actions=(1, 2, 3, 4, 5),
            key_fn=lambda e: (
                int(e.levels_completed),
                tuple((b.color, b.bbox, b.area)
                      for b in p.connected_components(e.frame(), (9, 10))),
            ),
            max_states=300,
            max_depth=60,
        )
        if pair_path is not None:
            return [(6, 31, 31), *small_path, (6, 46, 26), *pair_path]

        if len(small_path) < 40:
            for action in (1, 2, 3, 4):
                child = node.clone()
                child.step(action)
                q.append((child, small_path + [action]))
    return None


def revealed_small(node):
    reveal = node.clone()
    reveal.step(6, 46, 26)
    return blobs(reveal.frame(), (9,))


def small_agent_bfs(env):
    start = env.clone()
    start.step(6, 31, 31)

    def key(node):
        return (
            int(node.levels_completed),
            tuple(revealed_small(node)),
        )

    route = p.bounded_bfs(
        start,
        p.level_goal(env.levels_completed),
        actions=(1, 2, 3, 4, 5),
        key_fn=key,
        max_states=2000,
        max_depth=80,
    )
    return route


def probe(env):
    for name, route in (
        ("L1", legs.MIRRORED_PAIR_ASCENT),
        ("L2", legs.MIRRORED_PAIR_MAZE_REUNION),
    ):
        node = env.clone()
        if name == "L2":
            for action in legs.MIRRORED_PAIR_ASCENT:
                node.step(action)
        start_level = node.levels_completed
        print(name, "start", blobs(node.frame(), (9, 10, 12)))
        for index, action in enumerate(route, 1):
            before = blobs(node.frame(), (9, 10, 12))
            node.step(action)
            if node.levels_completed > start_level:
                print(name, "win_step", index, action, "before", before,
                      "after_level", node.levels_completed)
                break
    level3 = env.clone()
    for route in (legs.MIRRORED_PAIR_ASCENT, legs.MIRRORED_PAIR_MAZE_REUNION):
        for action in route:
            level3.step(action)
    print("L3_initial_counts", p.color_counts(level3.frame()))
    for point in ((31, 15), (11, 19), (39, 31), (2, 2)):
        selected = level3.clone()
        selected.step(6, *point)
        moved = selected.clone()
        moved.step(3)
        print("L3_select", point,
              "selected", blobs(selected.frame(), (1, 9, 10)),
              "move_delta", p.frame_delta(selected.frame(), moved.frame())["bbox"],
              "moved", blobs(moved.frame(), (1, 9, 10)))
    solve.solve(env)
    print("level", env.levels_completed + 1, "actions", env.actions)
    print("counts", p.color_counts(env.frame()))
    print("map")
    print("\n".join(coarse(env.frame())))
    print("objects", compact_objects(env.frame()))
    print("agents", agents(env.frame()), "goal", blobs(env.frame(), (9,)))
    print("closest", closest_pair(env))
    print("use_contexts", use_contexts(env))
    print("mixed_route", mixed_bfs(env))
    selected_small = env.clone()
    selected_small.step(6, 31, 31)
    for action in (1, 2, 3, 4):
        child = selected_small.clone()
        child.step(action)
        print("small_move", action, revealed_small(child))
    print("small_route", small_agent_bfs(env))
    cooperative_route = [1, 4, 4, (6, 31, 31), 3]
    cooperative = env.clone()
    for action in cooperative_route:
        if isinstance(action, tuple):
            cooperative.step(*action)
        else:
            cooperative.step(action)
    print("cooperative", cooperative_route, cooperative.levels_completed,
          cooperative.terminal(), p.color_counts(cooperative.frame()))
    coop_reveal = cooperative.clone()
    coop_reveal.step(6, 41, 21)
    print("cooperative_reveal", p.color_counts(coop_reveal.frame()),
          blobs(coop_reveal.frame(), (1, 9, 10)))
    staged = env.clone()
    for action in (1, 4, 4):
        staged.step(action)
    print("staged", agents(staged.frame()))
    for interaction in (5, (6, 31, 31), (6, 41, 21), (6, 26, 31)):
        node = staged.clone()
        if isinstance(interaction, tuple):
            node.step(*interaction)
        else:
            node.step(interaction)
        print("interact", interaction, "delta",
              p.frame_delta(staged.frame(), node.frame())["count"],
              "level", node.levels_completed, "agents", agents(node.frame()))
        for action in (1, 2, 3, 4, 5):
            child = node.clone()
            child.step(action)
            print("post_interact", interaction, action,
                  child.levels_completed, agents(child.frame()))
    for action, delta in p.action_deltas(env, env.actions).items():
        clone = env.clone()
        clone.step(action)
        print(
            "action", action, "delta", delta["count"], delta["bbox"],
            "small", blobs(clone.frame(), (9, 10)),
        )
    for point in ((46, 26), (16, 36), (31, 31)):
        selected = env.clone()
        selected.step(6, *point)
        print("selected", point, "level", selected.levels_completed,
              "terminal", selected.terminal(), p.color_counts(selected.frame()),
              blobs(selected.frame(), (1, 9, 10, 12)))
        effects = []
        for action in (1, 2, 3, 4, 5):
            child = selected.clone()
            child.step(action)
            effects.append((action, blobs(child.frame(), (1, 9, 10, 12)),
                            p.frame_delta(selected.frame(), child.frame())["count"]))
        print("select", point, "effects", effects)
    for point in ((11, 11), (26, 26), (36, 36), (41, 11), (11, 51)):
        selected = env.clone()
        selected.step(6, *point)
        effects = []
        for action in (1, 2, 3, 4):
            child = selected.clone()
            child.step(action)
            delta = p.frame_delta(selected.frame(), child.frame())
            effects.append((action, delta["count"], delta["bbox"],
                            p.color_counts(child.frame()).get(8, 0)))
        print("barrier_select", point, "effects", effects)
    for point in ((46, 26), (16, 36)):
        node = env.clone()
        node.step(6, 31, 31)
        node.step(6, *point)
        print("reselect", point, p.color_counts(node.frame()),
              blobs(node.frame(), (1, 9, 10)))
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            print("reselect_move", point, action,
                  blobs(child.frame(), (1, 9, 10)))
    for point in ((11, 11), (16, 11), (26, 26), (31, 26), (41, 11), (11, 51)):
        node = env.clone()
        node.step(6, 31, 31)
        node.step(6, *point)
        print("deselect_barrier", point, p.color_counts(node.frame()))
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            delta = p.frame_delta(node.frame(), child.frame())
            print("barrier_move", point, action, delta["count"], delta["bbox"],
                  p.color_counts(child.frame()))
    route = p.bounded_bfs(
        env,
        p.level_goal(env.levels_completed),
        actions=(1, 2, 3, 4, 5),
        key_fn=lambda e: (
            int(e.levels_completed),
            tuple((b.color, b.bbox, b.area)
                  for b in p.connected_components(e.frame(), (9, 10))),
        ),
        max_states=20000,
        max_depth=120,
    )
    print("route", route)
    if route:
        node = env.clone()
        trace = [agents(node.frame())]
        for action in route:
            node.step(action)
            trace.append(agents(node.frame()))
        print("trace", trace)
        print("won", node.levels_completed, node.terminal())


levels, path, err = arena.run_program("m0r0", probe)
print("result", levels, len(path), err)
