# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    walk(env, 2, 7)
    walk(env, 4, 4)
    rotate_quarter_turns(env, 3)


def play_level_2(env):
    walk(env, 1, 2)
    walk(env, 4, 4)

    select_figure(env, 23, 35)
    walk(env, 1, 8)
    walk(env, 4, 4)

    select_figure(env, 50, 53)
    rotate_quarter_turns(env, 3)
    walk(env, 1, 10)


def play_level_3(env):
    # Recovered from verified proposer path artifact: checkpoint.json+proposer_last.log
    for action in [[6, 35, 14], 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, [6, 35, 38], 5, 5, 3, 3, 3, 1, 1, 1, 1, 1, 1]:
        env.step(action)


def play_level_4(env):
    select_figure(env, 36, 21)
    rotate_quarter_turns(env, 1)
    walk(env, 2, 2)
    walk(env, 3, 7)

    select_figure(env, 45, 42)
    rotate_quarter_turns(env, 3)
    walk(env, 1, 4)
    walk(env, 3, 12)

    select_figure(env, 18, 48)
    rotate_quarter_turns(env, 1)
    walk(env, 1, 5)
    walk(env, 3, 3)


def play_level_5(env):
    from perception import arr, color_counts

    diagnostic = (
        [5] * 4 + [1] * 3 + [4] * 6
        + [(6, 54, 6)] + [5] * 3
        + [(6, 5, 38)] + [1] * 10 + [4] * 10
        + [(6, 47, 47)] + [5] * 3 + [1] * 10 + [4] * 2
    )

    def physical_summary(node):
        frame = arr(node.frame())
        counts = color_counts(frame)
        background = max(counts, key=counts.get)
        mask = frame != background
        mask[0, :] = False
        remaining = set(zip(*mask.nonzero()))
        sizes = []
        while remaining:
            seed = remaining.pop()
            todo = [seed]
            size = 1
            while todo:
                row, col = todo.pop()
                for near in ((row - 1, col), (row + 1, col),
                             (row, col - 1), (row, col + 1)):
                    if near in remaining:
                        remaining.remove(near)
                        todo.append(near)
                        size += 1
            sizes.append(size)
        return int(mask.sum()), sorted(sizes, reverse=True), counts

    probe = env.clone()
    print("L5_METRIC", 0, physical_summary(probe))
    for index, action in enumerate(diagnostic, 1):
        if isinstance(action, tuple):
            probe.step(*action)
        else:
            probe.step(action)
        if index in (13, 17, 18, 28, 38, 42, 48, 49, 50, 51, 52, 53, 54):
            print("L5_METRIC", index, action, probe.levels_completed,
                  physical_summary(probe))

    rotate_quarter_turns(env, 4)
    walk(env, 1, 3)
    walk(env, 4, 6)

    select_figure(env, 54, 6)
    rotate_quarter_turns(env, 3)

    select_figure(env, 5, 38)
    walk(env, 1, 10)
    walk(env, 4, 10)

    select_figure(env, 47, 47)
    rotate_quarter_turns(env, 3)
    walk(env, 1, 10)
    walk(env, 4, 2)


def play_level_6(env):
    """Temporary exact-junction clone search."""
    from perception import arr, bounded_bfs, color_counts

    def motion_key(state):
        pixels = arr(state.frame()).copy()
        pixels[0, :] = 0
        return pixels.tobytes()

    for index, click in enumerate(
        ((8, 8), (49, 7), (28, 22), (7, 40), (37, 46))
    ):
        variant = env.clone()
        variant.step(6, *click)
        for turns in range(5):
            counts = color_counts(variant.frame())
            background = max(counts, key=counts.get)
            cells = {
                (row // 3, col // 3): int(variant.frame()[row, col])
                for row in range(3, 64, 3)
                for col in range(0, 64, 3)
                if int(variant.frame()[row, col]) not in (4, background)
            }
            r0 = min(row for row, _ in cells)
            c0 = min(col for _, col in cells)
            shape = sorted((row - r0, col - c0) for row, col in cells)
            green = sorted(
                (row - r0, col - c0)
                for (row, col), value in cells.items()
                if value == 8
            )
            print("L6_VARIANT", index, turns, (r0, c0), shape, green)
            variant.step(5)
    return

    node = env.clone()
    node.step(6, 28, 22)
    for direction in (1, 2, 3, 4):
        for steps in (0, 2, 5, 10):
            probe = env.clone()
            probe.step(6, 28, 22)
            for _ in range(steps):
                probe.step(direction)
            before = probe.frame().copy()
            probe.step(5)
            changed = int((before != probe.frame()).sum())
            marks = [
                (row // 3, col // 3)
                for row in range(0, 64, 3)
                for col in range(0, 64, 3)
                if int(probe.frame()[row, col]) == 8
            ]
            settled = probe.clone()
            settled.step(1)
            settled_marks = [
                (row // 3, col // 3)
                for row in range(0, 64, 3)
                for col in range(0, 64, 3)
                if int(settled.frame()[row, col]) == 8
            ]
            print("L6_P2_TURN", direction, steps, changed, marks,
                  "then_up", settled_marks)
    repeated = env.clone()
    repeated.step(6, 28, 22)
    for turns in range(5):
        counts = color_counts(repeated.frame())
        print("L6_P2_REPEAT", turns, counts, [
            (row // 3, col // 3)
            for row in range(0, 64, 3)
            for col in range(0, 64, 3)
            if int(repeated.frame()[row, col]) == 8
        ])
        repeated.step(5)
    return
    path = bounded_bfs(
        node,
        lambda state, _: (
            int(state.frame()[6, 24]) == 8
            and int(state.frame()[12, 30]) == 8
        ),
        key_fn=motion_key,
        max_states=5000,
        max_depth=32,
    )
    print("L6_ORIENT_P2", path)
    if path is None:
        return
    for action in path:
        node.step(action)

    node.step(6, 6, 6)
    for _ in range(4):
        node.step(4)

    node.step(6, 49, 7)
    node.step(2)
    for _ in range(10):
        node.step(3)

    searches = (
        ("p3", (7, 40), lambda state: int(state.frame()[6, 24]) == 3),
        ("p4", (37, 46), lambda state: (
            state.levels_completed >= 6
            or color_counts(state.frame()).get(3, 0) >= 27
        )),
    )
    for label, click, target in searches:
        node.step(6, *click)
        if label in ("p3", "p4"):
            node.step(5)
        path = bounded_bfs(
            node,
            lambda state, _: target(state),
            key_fn=motion_key,
            max_states=3500,
            max_depth=40,
        )
        print("L6_TARGET", label, "start", color_counts(node.frame()),
              "path", path)
        if path is None:
            break
        for action in path:
            node.step(action)
        print("L6_TARGET_RESULT", label, node.levels_completed,
              color_counts(node.frame()), "marks", [
                  (row // 3, col // 3, int(node.frame()[row, col]))
                  for row in range(0, 64, 3)
                  for col in range(0, 64, 3)
                  if int(node.frame()[row, col]) in (3, 8)
              ])


def play_level_6(env):
    """Replay the exact paired-landmark layout."""
    from perception import color_counts

    def report(label):
        print("L6_REPLAY", label, color_counts(env.frame()), [
            (row // 3, col // 3, int(env.frame()[row, col]))
            for row in range(0, 64, 3)
            for col in range(0, 64, 3)
            if int(env.frame()[row, col]) in (3, 8)
        ])

    rotate_quarter_turns(env, 3)
    report("p0")

    select_figure(env, 49, 7)
    rotate_quarter_turns(env, 1)
    walk(env, 2, 2)
    walk(env, 3, 11)
    report("p1")

    select_figure(env, 7, 40)
    walk(env, 1, 11)
    walk(env, 4, 2)
    report("p3")

    select_figure(env, 28, 22)
    walk(env, 3, 4)
    report("p2")

    select_figure(env, 37, 46)
    walk(env, 1, 9)
    walk(env, 3, 7)
    report("p4")
    for label, click in (
        ("check_p0", (6, 6)),
        ("check_p1", (21, 12)),
        ("check_p3", (12, 9)),
        ("check_p2", (12, 24)),
        ("check_p4", (21, 21)),
    ):
        check = env.clone()
        check.step(6, *click)
        print("L6_CHECK", label, color_counts(check.frame()), [
            (row // 3, col // 3, int(check.frame()[row, col]))
            for row in range(0, 64, 3)
            for col in range(0, 64, 3)
            if int(check.frame()[row, col]) in (3, 8)
        ])
    select_figure(env, 16, 19)
    select_figure(env, 16, 19)
    select_figure(env, 16, 19)


def play_level_6(env):
    """Temporary component-merging clone search."""
    from perception import arr, bounded_bfs

    def components(node):
        pixels = arr(node.frame())
        values, counts = __import__("numpy").unique(
            pixels, return_counts=True
        )
        background = int(values[counts.argmax()])
        mask = pixels != background
        mask[0, :] = False
        remaining = set(zip(*mask.nonzero()))
        groups = []
        while remaining:
            seed = remaining.pop()
            todo = [seed]
            group = [seed]
            while todo:
                row, col = todo.pop()
                for near in ((row - 1, col), (row + 1, col),
                             (row, col - 1), (row, col + 1)):
                    if near in remaining:
                        remaining.remove(near)
                        todo.append(near)
                        group.append(near)
            groups.append(group)
        return sorted(groups, key=len, reverse=True)

    def key(node):
        pixels = arr(node.frame()).copy()
        pixels[0, :] = 0
        return pixels.tobytes()

    work = env.clone()
    whole = []
    for stage in range(6):
        base = len(components(work))
        print("L6_MERGE_STAGE", stage, base,
              [len(group) for group in components(work)])
        if base <= 1 or work.levels_completed >= 6:
            break
        samples = []
        for group in components(work):
            row, col = group[len(group) // 2]
            samples.append((col, row))
        progress = None
        for click in [None] + samples:
            root = work.clone()
            prefix = []
            if click is not None:
                root.step(6, *click)
                prefix = [(6, *click)]
            path = bounded_bfs(
                root,
                lambda node, _: (
                    node.levels_completed >= 6
                    or len(components(node)) < base
                ),
                key_fn=key,
                max_states=3000,
                max_depth=28,
            )
            print("L6_MERGE_TRY", click, path)
            if path is not None:
                progress = prefix + path
                break
        if progress is None:
            break
        for action in progress:
            if isinstance(action, tuple):
                work.step(*action)
            else:
                work.step(action)
        whole += progress
    print("L6_MERGE_PREFIX", whole)
    if len(components(work)) == 1 and work.levels_completed < 6:
        pixels = arr(work.frame())
        values, counts = __import__("numpy").unique(
            pixels, return_counts=True
        )
        background = int(values[counts.argmax()])
        roots = {key(work): (None, work.clone())}
        for row in range(3, 64, 3):
            for col in range(0, 64, 3):
                if int(pixels[row, col]) == background:
                    continue
                root = work.clone()
                root.step(6, col, row)
                roots.setdefault(key(root), ((col, row), root))
        print("L6_FINISH_ROOTS", len(roots), [
            click for click, _ in roots.values()
        ])
        for click, root in roots.values():
            prefix = [] if click is None else [(6, *click)]
            path = bounded_bfs(
                root,
                lambda node, _: node.levels_completed >= 6,
                key_fn=key,
                max_states=1500,
                max_depth=18,
            )
            print("L6_FINISH_TRY", click, path)
            if path is not None:
                print("L6_FOUND", whole + prefix + path)
                return


def play_level_6(env):
    """Temporary greedy reuse of the level-5 push-and-register policy."""
    from perception import arr, color_counts

    def metrics(node):
        pixels = arr(node.frame())
        counts = color_counts(pixels)
        background = max(counts, key=counts.get)
        mask = pixels != background
        mask[0, :] = False
        remaining = set(zip(*mask.nonzero()))
        groups = 0
        while remaining:
            groups += 1
            todo = [remaining.pop()]
            while todo:
                row, col = todo.pop()
                for near in ((row - 1, col), (row + 1, col),
                             (row, col - 1), (row, col + 1)):
                    if near in remaining:
                        remaining.remove(near)
                        todo.append(near)
        return groups, int(mask.sum())

    def push(node, direction):
        path = []
        for _ in range(24):
            before = arr(node.frame()).copy()
            level = node.levels_completed
            node.step(direction)
            if node.levels_completed > level:
                path.append(direction)
                return path, True
            changed = before != arr(node.frame())
            changed[0, :] = False
            if not changed.any():
                break
            path.append(direction)
        return path, False

    work = env.clone()
    whole = []
    for label, click in (
        ("p0", None),
        ("p1", (49, 7)),
        ("p2", (24, 27)),
        ("p3", (7, 47)),
        ("p4", (37, 46)),
    ):
        root = work.clone()
        prefix = []
        if click is not None:
            root.step(6, *click)
            prefix = [(6, *click)]
        choices = []
        for turns in range(5):
            node = root.clone()
            for _ in range(turns):
                node.step(5)
            start_area = metrics(node)[1]
            up, won = push(node, 1)
            right, won_right = ([], False) if won else push(node, 4)
            path = prefix + [5] * turns + up + right
            groups, area = metrics(node)
            choices.append((
                0 if (won or won_right) else 1,
                groups,
                max(0, start_area - area),
                len(path),
                turns,
                area,
                path,
                node,
            ))
            print("L6_PUSH_OPTION", label, turns,
                  (groups, area, max(0, start_area - area)),
                  node.levels_completed)
        choice = min(choices, key=lambda item: item[:5])
        whole += choice[6]
        work = choice[7]
        print("L6_PUSH_CHOICE", label, choice[:6])
        if work.levels_completed >= 6:
            print("L6_PUSH_FOUND", whole)
            return
    print("L6_PUSH_PREFIX", whole, metrics(work))


def play_level_6(env):
    """Temporary beam search over the reusable turn/push policy."""
    from perception import arr, color_counts

    def metrics(node):
        pixels = arr(node.frame())
        counts = color_counts(pixels)
        background = max(counts, key=counts.get)
        mask = pixels != background
        mask[0, :] = False
        remaining = set(zip(*mask.nonzero()))
        groups = 0
        while remaining:
            groups += 1
            todo = [remaining.pop()]
            while todo:
                row, col = todo.pop()
                for near in ((row - 1, col), (row + 1, col),
                             (row, col - 1), (row, col + 1)):
                    if near in remaining:
                        remaining.remove(near)
                        todo.append(near)
        return groups, int(mask.sum())

    def push(node, direction):
        path = []
        for _ in range(24):
            before = arr(node.frame()).copy()
            level = node.levels_completed
            node.step(direction)
            path.append(direction)
            if node.levels_completed > level:
                return path, True
            changed = before != arr(node.frame())
            changed[0, :] = False
            if not changed.any():
                path.pop()
                break
        return path, False

    def frame_key(node):
        pixels = arr(node.frame()).copy()
        pixels[0, :] = 0
        return pixels.tobytes()

    beam = [(env.clone(), [], 0)]
    clicks = (None, (49, 7), (24, 27), (7, 47), (37, 46))
    for stage, click in enumerate(clicks):
        expanded = {}
        for state, path, cumulative_loss in beam:
            root = state.clone()
            prefix = []
            if click is not None:
                root.step(6, *click)
                prefix = [(6, *click)]
            for turns in range(5):
                node = root.clone()
                for _ in range(turns):
                    node.step(5)
                start_area = metrics(node)[1]
                up, won = push(node, 1)
                right, won_right = ([], False) if won else push(node, 4)
                candidate_path = (
                    path + prefix + [5] * turns + up + right
                )
                if won or won_right or node.levels_completed >= 6:
                    print("L6_BEAM_FOUND", candidate_path)
                    return
                groups, area = metrics(node)
                loss = cumulative_loss + max(0, start_area - area)
                candidate = (node, candidate_path, loss)
                key = frame_key(node)
                prior = expanded.get(key)
                if prior is None or loss < prior[2]:
                    expanded[key] = candidate
        candidates = list(expanded.values())
        candidates.sort(key=lambda item: (
            metrics(item[0])[0],
            item[2],
            -metrics(item[0])[1],
            len(item[1]),
        ))
        beam = candidates
        print("L6_BEAM_STAGE", stage, len(beam), [
            (*metrics(state), loss) for state, _, loss in beam[:12]
        ])
    print("L6_BEAM_DONE", [
        (*metrics(state), loss, path) for state, path, loss in beam[:10]
    ])
