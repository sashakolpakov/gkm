from collections import deque

from perception import action_deltas, color_counts, connected_components


def _core(frame):
    cores = connected_components(frame, colors=(12,), min_area=4)
    return tuple(blob.bbox for blob in cores)


def _pegs(frame):
    return tuple(
        blob.bbox
        for blob in connected_components(frame, colors=(14,), min_area=4)
    )


def probe(env):
    frame = env.frame()
    blobs = [
        (b.color, b.bbox, b.area)
        for b in connected_components(frame, min_area=4)
    ]
    deltas = {
        action: (delta["count"], delta["bbox"])
        for action, delta in action_deltas(env, actions=(1, 2, 3, 4)).items()
    }
    print("L2 actions", env.actions)
    print("L2 colors", color_counts(frame))
    print("L2 blobs", blobs)
    print("L2 key_deltas", deltas)
    queue = deque([(env.clone(), ())])
    seen = {_core(frame)}
    transitions = []
    while queue and len(seen) < 120:
        node, path = queue.popleft()
        here = _core(node.frame())
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            there = _core(child.frame())
            if there != here:
                transitions.append((here, action, there, _pegs(child.frame())))
            if there not in seen:
                seen.add(there)
                queue.append((child, path + (action,)))
    print("L2 reachable_cores", sorted(seen))
    print("L2 transitions", transitions[:40])
    start_level = env.levels_completed
    queue = deque([(env.clone(), ())])
    seen_states = {(_core(frame), _pegs(frame))}
    peg_moves = []
    solution = None
    while queue and len(seen_states) < 1200:
        node, path = queue.popleft()
        if node.levels_completed > start_level:
            solution = path
            break
        if len(path) >= 60:
            continue
        before_pegs = _pegs(node.frame())
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            key = (_core(child.frame()), _pegs(child.frame()))
            if key[1] != before_pegs:
                peg_moves.append((path, action, before_pegs, key[1],
                                  child.levels_completed))
            if key not in seen_states:
                seen_states.add(key)
                queue.append((child, path + (action,)))
    print("L2 peg_moves", peg_moves[:20])
    print("L2 movement_solution", solution, "states", len(seen_states))
    click_clone = env.clone()
    captures = (
        ((15, 13), (15, 25)),
        ((15, 25), (15, 37)),
        ((15, 43), (15, 31)),
    )
    click_progress = []
    for source, destination in captures:
        click_clone.step(6, source[1] + 1, source[0] + 1)
        click_clone.step(6, destination[1] + 1, destination[0] + 1)
        click_progress.append((_pegs(click_clone.frame()),
                               click_clone.levels_completed))
    print("L2 capture_progress", click_progress)
    print("L2 post_capture", {
        "terminal": click_clone.terminal(),
        "level": click_clone.levels_completed,
        "colors": color_counts(click_clone.frame()),
        "core": _core(click_clone.frame()),
        "objects": [
            (b.color, b.bbox, b.area)
            for b in connected_components(
                click_clone.frame(), colors=(1, 2, 9, 11, 12, 14, 15), min_area=4
            )
        ],
    })
    queue = deque([(click_clone.clone(), ())])
    seen = {_core(click_clone.frame())}
    post_solution = None
    while queue and len(seen) < 200:
        node, path = queue.popleft()
        if node.levels_completed > start_level:
            post_solution = path
            break
        if len(path) >= 60:
            continue
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            key = _core(child.frame())
            if child.levels_completed > start_level:
                post_solution = path + (action,)
                queue.clear()
                break
            if key not in seen:
                seen.add(key)
                queue.append((child, path + (action,)))
    print("L2 post_movement_solution", post_solution, "cores", sorted(seen))
    def agents(frame):
        return tuple(
            (b.color, b.bbox, b.area)
            for b in connected_components(frame, colors=(12, 15), min_area=4)
        )
    post_effects = []
    for action in (1, 2, 3, 4):
        child = click_clone.clone()
        child.step(action)
        post_effects.append((action, agents(child.frame()),
                             child.levels_completed))
    print("L2 post_key_effects", post_effects)
    for action in (1, 2, 3, 4):
        child = click_clone.clone()
        timeline = []
        for _ in range(12):
            child.step(action)
            timeline.append((agents(child.frame()), child.levels_completed))
            if child.levels_completed > start_level:
                break
        print("L2 repeat", action, timeline)
    click_tests = ((32, 16), (44, 52), (50, 46), (50, 52),
                   (7, 57), (32, 34))
    click_effects = []
    for x, y in click_tests:
        child = click_clone.clone()
        before = child.frame()
        child.step(6, x, y)
        delta = action_deltas.__globals__["frame_delta"](before, child.frame())
        click_effects.append(((x, y), (delta["count"], delta["bbox"]),
                              color_counts(child.frame()),
                              child.levels_completed))
    print("L2 post_click_effects", click_effects)
    f = click_clone.frame()
    chars = {9: "#", 15: "X"}
    print("L2 color15_mask")
    for row in f[50:64, 0:14]:
        print("".join(chars.get(int(v), ".") for v in row))
    queue = deque([(env.clone(), ())])
    position_paths = {_core(env.frame()): ()}
    while queue:
        node, path = queue.popleft()
        for action in (1, 2, 3, 4):
            child = node.clone()
            child.step(action)
            key = _core(child.frame())
            if key not in position_paths:
                position_paths[key] = path + (action,)
                queue.append((child, path + (action,)))
    staged = []
    for position, path in position_paths.items():
        node = env.clone()
        for action in path:
            node.step(action)
        for source, destination in captures:
            node.step(6, source[1] + 1, source[0] + 1)
            node.step(6, destination[1] + 1, destination[0] + 1)
        staged.append((position, path, node.levels_completed,
                       color_counts(node.frame()).get(15, 0)))
    print("L2 staged_completions", staged)
