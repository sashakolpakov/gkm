"""Shared skills inferred and verified through public frame/action observations."""
from perception import connected_components, safe_step


def wait_for_marker_color(env, target_color, idle_action=(6, 0, 10),
                          max_steps=32):
    """Let autonomous objects act until a field marker has the requested color.

    Verify the color subgoal on a clone before committing the waiting actions.
    The idle coordinate should be away from markers so it does not pull them.
    """
    trial = env.clone()
    base_level = env.levels_completed
    path = []
    for _ in range(max_steps + 1):
        if any(b.bbox[0] >= 10 for b in connected_components(
                trial.frame(), colors=[target_color], min_area=1)):
            for action in path:
                safe_step(env, action)
            return
        if (len(path) == max_steps or trial.terminal() or
                trial.levels_completed != base_level):
            return
        safe_step(trial, idle_action)
        path.append(idle_action)


def click_walk_to_region(env, marker_color=15, goal_color=9,
                         stride=6, max_clicks=32, goal_index=None,
                         arrival_radius=None):
    """Walk a field marker into a colored region with short coordinate clicks.

    Plan on a clone, measuring distance to the region's center. Commit only a
    path that increases the level reward, or reaches an explicitly requested
    arrival radius for a partial placement. Goal indices run left to right.
    Header objects are excluded from the movable marker candidates.
    """
    trial = env.clone()
    base_level = env.levels_completed
    goals = connected_components(trial.frame(), colors=[goal_color], min_area=4)
    markers = [b for b in connected_components(
        trial.frame(), colors=[marker_color], min_area=4) if b.bbox[0] >= 10]
    if not goals or not markers:
        return
    target = (max(goals, key=lambda b: b.area) if goal_index is None else
              sorted(goals, key=lambda b: b.centroid[1])[goal_index]).centroid
    marker = max(markers, key=lambda b: b.area).centroid
    path = []
    for _ in range(max_clicks):
        y, x = marker
        ty, tx = target
        nx = round(x + max(-stride, min(stride, tx - x)))
        ny = round(y + max(-stride, min(stride, ty - y)))
        action = (6, nx, ny)
        safe_step(trial, action)
        path.append(action)
        if trial.levels_completed > base_level:
            for action in path:
                safe_step(env, action)
                if env.levels_completed > base_level or env.terminal():
                    return
            return
        markers = [b for b in connected_components(
            trial.frame(), colors=[marker_color], min_area=4)
                   if b.bbox[0] >= 10]
        if not markers:
            return
        next_marker = min(markers, key=lambda b:
                          (b.centroid[0]-y)**2 + (b.centroid[1]-x)**2).centroid
        before = (ty-y)**2 + (tx-x)**2
        after = (ty-next_marker[0])**2 + (tx-next_marker[1])**2
        if arrival_radius is not None and after <= arrival_radius**2:
            for action in path:
                safe_step(env, action)
                if env.levels_completed > base_level or env.terminal():
                    return
            return
        if after >= before:
            return
        marker = next_marker


def merge_matching_markers(env, colors=(10, 6, 15, 11), stride=3,
                           max_clicks=96, target_count=1):
    """Bring nearest equal field markers together to a requested count.

    Short clicks move nearby markers; equal neighbors fuse into the next size.
    Plan on a clone and commit only after the marker-count subgoal is met.
    """
    from math import hypot

    def markers(world):
        return [b for b in connected_components(world.frame(), colors=colors)
                if b.bbox[0] >= 10]

    trial = env.clone()
    path = []
    for _ in range(max_clicks):
        blobs = markers(trial)
        if len(blobs) == target_count:
            for action in path:
                safe_step(env, action)
            return
        pairs = [(hypot(a.centroid[0] - b.centroid[0],
                        a.centroid[1] - b.centroid[1]), a, b)
                 for i, a in enumerate(blobs) for b in blobs[i + 1:]
                 if a.color == b.color]
        if not pairs:
            return
        _, source, target = min(pairs, key=lambda pair: pair[0])
        # The observed marker anchor is the lower-right pixel of its square.
        y, x = source.bbox[2:]
        ty, tx = target.bbox[2:]
        distance = hypot(ty - y, tx - x)
        step = min(stride, distance)
        action = (6, round(x + (tx - x) * step / (distance or 1)),
                  round(y + (ty - y) * step / (distance or 1)))
        before = [(b.color, b.bbox) for b in blobs]
        safe_step(trial, action)
        path.append(action)
        if [(b.color, b.bbox) for b in markers(trial)] == before:
            return


def merge_preserving_mass(env, colors=(10, 6, 15, 11), hazard_color=7,
                          goal_color=9, beam_width=12, max_depth=45,
                          target_color=None):
    """Merge equal markers while avoiding losses caused by moving hazards.

    Marker colors encode successive doublings. Search short coordinate pulls
    on clones, reject any loss of total mass, and favor fewer markers and
    shorter distances between equal pairs. Commit after merging to one marker
    or producing the requested target color while preserving total mass.
    """
    import math
    import time

    weights = {color: 2 ** i for i, color in enumerate(colors)}

    def markers(world):
        return [b for b in connected_components(world.frame(), colors=colors)
                if b.bbox[0] >= 10]

    def points(blobs):
        return [(b.color, b.bbox[3], b.bbox[2]) for b in blobs]

    goals = connected_components(env.frame(), colors=[goal_color], min_area=4)
    if not goals:
        return
    gy, gx = max(goals, key=lambda b: b.area).centroid

    def score(blobs):
        positions = points(blobs)
        distance = 0
        for color in colors:
            group = [(x, y) for c, x, y in positions if c == color]
            while len(group) > 1:
                d, i, j = min((math.dist(a, b), i, j)
                              for i, a in enumerate(group)
                              for j, b in enumerate(group) if j > i)
                distance += d
                group.pop(j)
                group.pop(i)
        return (len(blobs) * 100 + distance + .2 * sum(
            math.dist((x, y), (gx, gy)) * weights[c] / total
            for c, x, y in positions))

    trial = env.clone()
    total = sum(weights[b.color] for b in markers(trial))
    if not total:
        return
    beam = [(score(markers(trial)), trial, [])]
    seen = set()
    started = time.monotonic()
    probes = 0
    for _ in range(max_depth):
        successors = []
        for _, world, path in beam:
            positions = points(markers(world))
            actions = set()
            for c, x, y in positions:
                for dx, dy in [(0, 0), (6, 0), (-6, 0), (0, 6), (0, -6),
                               (4, 4), (-4, 4), (4, -4), (-4, -4)]:
                    if 1 <= x + dx <= 62 and 11 <= y + dy <= 61:
                        actions.add((6, x + dx, y + dy))
            for i, (c, x, y) in enumerate(positions):
                for other, tx, ty in positions[i + 1:]:
                    if c == other and math.dist((x, y), (tx, ty)) <= 18:
                        actions.add((6, round((x + tx) / 2),
                                     round((y + ty) / 2)))
            for action in sorted(actions):
                child = world.clone()
                safe_step(child, action)
                probes += 1
                if probes % 100 == 0:
                    delay = probes / 250 - (time.monotonic() - started)
                    if delay > 0:
                        time.sleep(delay)
                blobs = markers(child)
                if sum(weights[b.color] for b in blobs) != total:
                    continue
                hazard = tuple(b.bbox for b in connected_components(
                    child.frame(), colors=[hazard_color]))
                key = (tuple(points(blobs)), hazard)
                if key in seen:
                    continue
                seen.add(key)
                next_path = path + [action]
                if (len(blobs) == 1 or
                        target_color is not None and any(
                            b.color == target_color for b in blobs)):
                    for step in next_path:
                        safe_step(env, step)
                    return
                successors.append((score(blobs), child, next_path))
        successors.sort(key=lambda item: item[0])
        beam = successors[:beam_width]
        if not beam:
            return


def place_matching_markers(env, marker_color=11, goal_color=9,
                           beam_width=16, max_depth=18):
    """Route matching markers into distinct regions, rejecting marker losses.

    Use assignment distance as dense progress and clone each short pull to
    account for autonomous obstacles. Commit only a reward-verified path.
    """
    import itertools
    import math
    import time

    def markers(world):
        return [b for b in connected_components(world.frame(), colors=[marker_color])
                if b.bbox[0] >= 10]

    goals = [b.centroid for b in connected_components(
        env.frame(), colors=[goal_color], min_area=4)]
    count = len(markers(env))
    if not count or count > len(goals):
        return
    base = env.levels_completed

    def score(blobs):
        return min(sum(math.dist(b.centroid, g) for b, g in zip(blobs, order))
                   for order in itertools.permutations(goals, count))

    beam = [(score(markers(env)), env.clone(), [])]
    seen = set()
    started = time.monotonic()
    probes = 0
    for depth in range(max_depth):
        successors = []
        for _, world, path in beam:
            actions = {(6, 0, 10)}
            for b in markers(world):
                y, x = b.centroid
                for dx, dy in [(0,0),(6,0),(-6,0),(0,6),(0,-6),
                               (4,4),(-4,4),(4,-4),(-4,-4)]:
                    if 0 <= round(x+dx) <= 63 and 10 <= round(y+dy) <= 62:
                        actions.add((6,round(x+dx),round(y+dy)))
                for gy, gx in goals:
                    actions.add((6,round(x+max(-6,min(6,gx-x))),
                                 round(y+max(-6,min(6,gy-y)))))
            for action in sorted(actions):
                child = world.clone()
                safe_step(child, action)
                probes += 1
                if probes % 50 == 0:
                    time.sleep(max(0, probes/250-(time.monotonic()-started)))
                next_path = path + [action]
                if child.levels_completed > base:
                    for step in next_path:
                        safe_step(env, step)
                    return
                blobs = markers(child)
                if len(blobs) != count:
                    continue
                key = bytes(memoryview(child.frame()))
                if key in seen:
                    continue
                seen.add(key)
                successors.append((score(blobs), child, next_path))
        successors.sort(key=lambda item: item[0])
        beam = successors[:beam_width]
        if not beam:
            return


def reduce_markers_for_placement(env, oversized_color, target_color,
                                 goal_color=9, beam_width=20, max_depth=18):
    """Expose oversized markers to moving hazards while preserving target peers.

    Search short pulls on clones. Keep the marker count fixed, accept only the
    two requested sizes, and use distance to distinct regions as dense progress.
    Commit when every marker has the target size, leaving final placement to
    place_matching_markers.
    """
    import itertools
    import math
    import time

    def markers(world):
        return [b for b in connected_components(world.frame(),
                colors=[oversized_color, target_color]) if b.bbox[0] >= 10]

    goals = [b.centroid for b in connected_components(
        env.frame(), colors=[goal_color], min_area=4)]
    count = len(markers(env))
    if not count or count > len(goals):
        return
    base = env.levels_completed

    def score(blobs):
        distance = min(sum(math.dist(b.centroid, g)
                           for b, g in zip(blobs, order))
                       for order in itertools.permutations(goals, count))
        return distance + 10 * sum(b.color == oversized_color for b in blobs)

    beam = [(score(markers(env)), env.clone(), [])]
    seen = set()
    probes = 0
    started = time.monotonic()
    for _ in range(max_depth):
        successors = []
        for _, world, path in beam:
            actions = {(6, 0, 10)}
            for b in markers(world):
                y, x = b.centroid
                for dx, dy in [(0,0),(6,0),(-6,0),(0,6),(0,-6),
                               (4,4),(-4,4),(4,-4),(-4,-4)]:
                    nx, ny = round(x + dx), round(y + dy)
                    if 0 <= nx <= 63 and 10 <= ny <= 62:
                        actions.add((6, nx, ny))
            for action in sorted(actions):
                child = world.clone()
                safe_step(child, action)
                probes += 1
                if probes % 50 == 0:
                    time.sleep(max(0, probes / 250 -
                                   (time.monotonic() - started)))
                blobs = markers(child)
                if len(blobs) != count:
                    continue
                next_path = path + [action]
                if (child.levels_completed > base or
                        all(b.color == target_color for b in blobs)):
                    for step in next_path:
                        safe_step(env, step)
                    return
                key = bytes(memoryview(child.frame()))
                if key in seen:
                    continue
                seen.add(key)
                successors.append((score(blobs), child, next_path))
        successors.sort(key=lambda item: item[0])
        beam = successors[:beam_width]
        if not beam:
            return


def field_objects(env):
    """Locate square markers and diagonal moving shapes, including merge flashes."""
    import numpy as np
    from scipy.ndimage import label, find_objects

    frame = np.asarray(env.frame())
    result = []
    for color in (10, 6, 15, 11, 12, 8):
        mask = frame == color
        mask[:10] = False
        labels, _ = label(mask, structure=np.ones((3, 3)))
        for region in find_objects(labels):
            if region:
                ys, xs = region
                result.append((color, (xs.start + xs.stop - 1) / 2,
                               (ys.start + ys.stop - 1) / 2))
    # Diagonal sprites need templates: adjacent moving sprites can touch without
    # merging, so a connected-component count alone loses their identities.
    patterns = {
        7: [(-2, 0), (-1, -1), (-1, 1), (0, -2), (0, 2),
            (1, -1), (1, 0), (1, 1)],
        14: [(-2, 0), (-1, -1), (-1, 1), (0, -2), (0, 0), (0, 2),
             (1, -1), (1, 1)],
        13: [(-2, -2), (-2, 0), (-2, 2), (-1, -1), (-1, 1),
             (0, -2), (0, -1), (0, 0), (0, 1), (0, 2),
             (1, -1), (1, 0), (1, 1)],
    }
    for color, pixels in patterns.items():
        hits = np.ones((51, 60), dtype=bool)
        for dy, dx in pixels:
            hits &= frame[12 + dy:63 + dy, 2 + dx:62 + dx] == color
        if color == 13:
            flash = np.ones((51, 60), dtype=bool)
            for dy, dx in pixels:
                flash &= frame[12 + dy:63 + dy, 2 + dx:62 + dx] == 2
            for y, x in np.argwhere(flash):
                if np.count_nonzero(frame[y + 10:y + 14, x:x + 5] == 2) == 13:
                    hits[y, x] = True
        for y, x in np.argwhere(hits):
            result.append((color, float(x + 2), float(y + 12)))
    return result


def assemble_mixed_objectives(env, marker_targets, agent_target_color,
                              marker_colors=(10, 6, 15, 11, 12, 8),
                              agent_colors=(7, 14, 13), beam_width=32,
                              max_depth=24):
    """Merge moving objects and place mixed marker sizes in distinct regions.

    Keep enough marker mass for each requested size while allowing moving
    objects to trim oversized markers. Dense progress combines feasible mass
    partitions, equal-agent separation, and marker assignment distance. Only
    commit a path after a clone increases the real level reward.
    """
    import itertools
    import math
    import time

    weights = {c: 2 ** i for i, c in enumerate(marker_colors)}
    agent_weights = {c: 2 ** i for i, c in enumerate(agent_colors)}
    goals = [(b.centroid[1], b.centroid[0]) for b in connected_components(
        env.frame(), colors=[9], min_area=4)]
    target_mass = [weights[c] for c in marker_targets]
    base_level = env.levels_completed

    def score(obs):
        agents = [p for p in obs if p[0] in agent_colors]
        squares = [p for p in obs if p[0] in marker_colors]
        if (len(squares) < len(target_mass) or len(squares) > len(goals)
                or sum(agent_weights[p[0]] for p in agents)
                != agent_weights[agent_target_color]):
            return None
        excess = []
        for assignment in itertools.product(range(len(target_mass)),
                                             repeat=len(squares)):
            totals = [sum(weights[p[0]] for p, group in zip(squares, assignment)
                          if group == i) for i in range(len(target_mass))]
            if all(m >= required for m, required in zip(totals, target_mass)):
                excess.append(sum(m > required for m, required in
                                  zip(totals, target_mass)))
        if not excess:
            return None
        value = (18 * (len(agents) - 1)
                 + 20 * (len(squares) - len(target_mass)) + 100 * min(excess))
        for color in (*agent_colors[:-1], *marker_colors):
            peers = [p for p in obs if p[0] == color]
            if len(peers) >= 2:
                value += .3 * min(math.dist(a[1:], b[1:])
                                  for a, b in itertools.combinations(peers, 2))
        value += 2 * min(sum(math.dist(p[1:], g) for p, g in zip(squares, order))
                         for order in itertools.permutations(goals, len(squares)))
        return value

    initial = score(field_objects(env))
    if initial is None or not goals:
        return
    beam = [(initial, env.clone(), [])]
    seen = set()
    probes = 0
    started = time.monotonic()
    for _ in range(max_depth):
        successors = []
        for _, world, path in beam:
            obs = field_objects(world)
            actions = {(6, 0, 10)}
            for color, x, y in obs:
                points = [(x + dx, y + dy) for dx, dy in
                          [(0, 0), (6, 0), (-6, 0), (0, 6), (0, -6),
                           (4, 4), (-4, 4), (4, -4), (-4, -4)]]
                points += [(x + max(-6, min(6, gx - x)),
                            y + max(-6, min(6, gy - y))) for gx, gy in goals]
                for nx, ny in points:
                    nx, ny = round(nx), round(ny)
                    if 0 <= nx <= 63 and 10 <= ny <= 62:
                        actions.add((6, nx, ny))
            for a, b in itertools.combinations(obs, 2):
                if a[0] == b[0] and math.dist(a[1:], b[1:]) <= 20:
                    actions.add((6, round((a[1] + b[1]) / 2),
                                 round((a[2] + b[2]) / 2)))
            for action in sorted(actions):
                child = world.clone()
                try:
                    safe_step(child, action)
                except IndexError:
                    # The public arena can reject a state with no surviving marker.
                    continue
                probes += 1
                if probes % 50 == 0:
                    time.sleep(max(0, probes / 250 - (time.monotonic() - started)))
                next_path = path + [action]
                if child.levels_completed > base_level:
                    for step in next_path:
                        safe_step(env, step)
                    return
                value = score(field_objects(child))
                if value is None:
                    continue
                key = bytes(memoryview(child.frame()))
                if key in seen:
                    continue
                seen.add(key)
                successors.append((value, child, next_path))
        successors.sort(key=lambda item: item[0])
        beam = successors[:beam_width]
        if not beam:
            return
