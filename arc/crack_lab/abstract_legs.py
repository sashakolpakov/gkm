"""Abstract legs as OBSERVATION-POLICIES (channel-blind), not move-sequences, so
they transfer across levels. Each leg re-instantiates on the current frame via a
learned game-specific binding (built from exploration). No move1->move10 rigidity.

Legs:
  seek(target)             : move the avatar to a target object
  push_to(box, region)     : push a box into the region (deliver)
  displace(box, away)      : push a box AWAY to clear a pass (the non-monotone
                             "move a box back" leg — the L2 enabler)
Bindings (game-specific, learned): action->direction sign; which colour is
avatar / box-carrier / region / wall. The policies read these; the engine
composes legs and verifies via the real step.
"""
from __future__ import annotations
import copy, heapq
import numpy as np
from lab import arc, make_env
from logical_grid import Grid, objects
from arcengine import ActionInput, GameAction as EA

NAME = {0: "RESET", 1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4", 5: "ACTION5"}


def arr_of(fd):
    return np.asarray(fd.frame[-1])


def step(g, a):
    gc = copy.deepcopy(g)
    return gc, gc.perform_action(ActionInput(id=EA[NAME[a]]), raw=True)


def comps_centroids(arr, color, exclude=None):
    out = []
    for c in arc.connected_components([list(r) for r in arr], color):
        cx = sum(p[0] for p in c) / len(c); cy = sum(p[1] for p in c) / len(c)
        out.append((cx, cy))
    return out


def bbox_center(arr, color):
    ys, xs = np.where(arr == color)
    if len(xs) == 0:
        return None
    return ((int(xs.min()) + int(xs.max())) / 2, (int(ys.min()) + int(ys.max())) / 2)


def learn_dirs(g0):
    """Binding: action -> unit direction (sign), learned by probing the avatar."""
    dirs = {}
    base = bbox_center(arr_of(copy.deepcopy(g0).perform_action(ActionInput(id=EA.RESET), raw=True)), 14)
    for a in (1, 2, 3, 4):
        gc = copy.deepcopy(g0)
        gc.perform_action(ActionInput(id=EA.RESET), raw=True)
        fd = gc.perform_action(ActionInput(id=EA[NAME[a]]), raw=True)
        c = bbox_center(arr_of(fd), 14)
        if c and base:
            dx, dy = c[0] - base[0], c[1] - base[1]
            if abs(dx) >= abs(dy):
                dirs[a] = (1 if dx > 0 else -1, 0)
            else:
                dirs[a] = (0, 1 if dy > 0 else -1)
    return dirs


def nav_action(av, target, dirs):
    """Pick the action whose direction reduces the larger axis gap to target."""
    dx, dy = target[0] - av[0], target[1] - av[1]
    want = (1 if dx > 0 else -1, 0) if abs(dx) >= abs(dy) else (0, 1 if dy > 0 else -1)
    for a, d in dirs.items():
        if d == want:
            return a
    return None


def action_for_dir(dirs, vec):
    return next((act for act, d in dirs.items() if d == vec), None)


def dominant_dir(src, dst):
    dx, dy = dst[0] - src[0], dst[1] - src[1]
    return (1 if dx > 0 else -1, 0) if abs(dx) >= abs(dy) else (0, 1 if dy > 0 else -1)


def box_progress(before, after, goal):
    return abs(before[0] - goal[0]) + abs(before[1] - goal[1]) - abs(after[0] - goal[0]) - abs(after[1] - goal[1])


def is_game_over(fd):
    return str(fd.state).endswith("GAME_OVER")


def locked_pixels(arr, carrier, footprint):
    return sum(1 for x, y in footprint if arr[y][x] == carrier)


def component_cells(arr, grid, carrier):
    return [o.cell for o in objects(arr, grid, [carrier])]


def track_cell(arr, grid, carrier, previous, max_jump=2):
    cells = component_cells(arr, grid, carrier)
    if not cells:
        return None
    cell = min(cells, key=lambda c: abs(c[0] - previous[0]) + abs(c[1] - previous[1]))
    jump = abs(cell[0] - previous[0]) + abs(cell[1] - previous[1])
    return cell if jump <= max_jump else None


def nearest_footprint_cell(grid, footprint, cell):
    fp_cells = {grid.cell_of_pixel(x, y) for x, y in footprint}
    return min(fp_cells, key=lambda c: abs(c[0] - cell[0]) + abs(c[1] - cell[1]))


def searched_push_to(g0, fd0, target0, carrier, footprint, dirs, avatar=14,
                     max_steps=60, max_nodes=12000):
    """Find one delivery on cloned real states; never mutates the caller's branch.

    The selected component is tracked by logical-cell continuity. A branch
    succeeds only when carrier pixels newly occupy the original region footprint
    or the game reports a level transition.
    """
    start = arr_of(fd0)
    grid = Grid.infer(start)
    seed = grid.cell_of_pixel(round(target0[0]), round(target0[1]))
    target = track_cell(start, grid, carrier, seed, max_jump=3)
    avatar_cell = next((o.cell for o in objects(start, grid, [avatar])), None)
    if target is None or avatar_cell is None:
        return ("blocked", g0, fd0, [])

    goal = nearest_footprint_cell(grid, footprint, target)
    locked0 = locked_pixels(start, carrier, footprint)
    level0 = fd0.levels_completed
    actions = tuple(sorted(dirs))
    seen = {start.tobytes()}

    def heuristic(arr, box_cell):
        av = next((o.cell for o in objects(arr, grid, [avatar])), avatar_cell)
        box_d = abs(box_cell[0] - goal[0]) + abs(box_cell[1] - goal[1])
        av_d = abs(av[0] - box_cell[0]) + abs(av[1] - box_cell[1])
        return 4 * box_d + av_d

    heap = [(heuristic(start, target), 0, 0, g0, fd0, target, ())]
    counter = 1
    nodes = 0
    while heap and nodes < max_nodes:
        _, _, depth, g, fd, box_cell, path = heapq.heappop(heap)
        if depth >= max_steps:
            continue
        for action in actions:
            gc = copy.deepcopy(g)
            fdc = gc.perform_action(ActionInput(id=EA[NAME[action]]), raw=True)
            nodes += 1
            if fdc.levels_completed > level0:
                return ("win", gc, fdc, list(path) + [action])
            if is_game_over(fdc) or not getattr(fdc, "frame", None):
                continue
            arr = arr_of(fdc)
            if locked_pixels(arr, carrier, footprint) > locked0:
                return ("delivered", gc, fdc, list(path) + [action])
            key = arr.tobytes()
            if key in seen:
                continue
            seen.add(key)
            next_cell = track_cell(arr, grid, carrier, box_cell)
            if next_cell is None:
                continue
            next_path = path + (action,)
            score = heuristic(arr, next_cell) + 0.05 * len(next_path)
            heapq.heappush(heap, (score, counter, depth + 1, gc, fdc, next_cell, next_path))
            counter += 1
    return ("blocked", g0, fd0, [])


def push_box(g, fd, target0, goal_c, carrier, dirs, avatar=14, max_steps=60, path=None):
    """Closed-loop object-relative push toward an arbitrary goal point.

    Returns ('win'|'moved'|'reached'|'blocked', g, fd). This is the common leg
    used by both `push_to` (goal = region centre) and `displace_box` (goal = a
    point away from the region / blockage)."""
    level0 = fd.levels_completed
    stuck = 0
    moved_any = False
    for _ in range(max_steps):
        arr = arr_of(fd)
        bs = comps_centroids(arr, carrier)
        if not bs:
            return ("blocked", g, fd)
        B = min(bs, key=lambda c: abs(c[0] - target0[0]) + abs(c[1] - target0[1]))
        av = bbox_center(arr, avatar)
        if av is None:
            return ("blocked", g, fd)
        pdir = dominant_dir(B, goal_c)
        behind = (B[0] - pdir[0] * 6, B[1] - pdir[1] * 6)
        if abs(av[0] - behind[0]) + abs(av[1] - behind[1]) > 3:
            a = nav_action(av, behind, dirs)
        else:
            a = action_for_dir(dirs, pdir)
        if a is None:
            return ("blocked", g, fd)
        B_before = B
        if path is not None:
            path.append(a)
        g, fd = step(g, a)
        if fd.levels_completed > level0:
            return ("win", g, fd)
        if str(fd.state).endswith("GAME_OVER"):
            return ("blocked", g, fd)
        arr2 = arr_of(fd)
        bs2 = comps_centroids(arr2, carrier)
        if not bs2:
            return ("reached", g, fd)
        B2 = min(bs2, key=lambda c: abs(c[0] - target0[0]) + abs(c[1] - target0[1]))
        if box_progress(B_before, B2, goal_c) > 0:
            moved_any = True
            stuck = 0
            if abs(B2[0] - goal_c[0]) + abs(B2[1] - goal_c[1]) <= 3:
                return ("reached", g, fd)
        else:
            stuck += 1
            if stuck > 12:
                return ("moved", g, fd) if moved_any else ("blocked", g, fd)
    return ("moved", g, fd) if moved_any else ("blocked", g, fd)


def push_to(g, fd, target0, region_c, carrier, footprint, dirs, avatar=14, max_steps=60, path=None):
    """Closed-loop policy: push the box nearest target0 into the region. Returns
    ('win'|'delivered'|'blocked', g, fd). If `path` is given, every applied action
    is appended (so the whole plan is replay-validatable)."""
    status, gc, fdc, leg_path = searched_push_to(
        g, fd, target0, carrier, footprint, dirs, avatar=avatar, max_steps=max_steps)
    if status in ("win", "delivered"):
        if path is not None:
            path.extend(leg_path)
        return (status, gc, fdc)
    return ("blocked", g, fd)


def displace_box(g, fd, target0, away_from, carrier, dirs, avatar=14, distance=12, max_steps=40, path=None):
    """Push the chosen box away from `away_from` to clear a corridor / deadlock.

    This is intentionally non-monotone: it may temporarily worsen the delivery
    count to create a later admissible route."""
    goal_vec = dominant_dir(away_from, target0)
    goal_c = (target0[0] + goal_vec[0] * distance, target0[1] + goal_vec[1] * distance)
    return push_box(g, fd, target0, goal_c, carrier, dirs, avatar=avatar, max_steps=max_steps, path=path)


if __name__ == "__main__":   # validate: does push_to deliver ONE box on wa30 L1?
    import priors
    e = make_env("wa30")(); e.reset(); g0 = copy.deepcopy(e._env._game)
    fd = g0.perform_action(ActionInput(id=EA.RESET), raw=True)
    a0 = arr_of(fd)
    struct = set(priors.structure_colours(a0.tolist()))
    cont = [(r, i) for (r, i, _c) in priors.containers(a0.tolist()) if r not in struct and i not in struct]
    ring, region = cont[0]
    fp = [(x, y) for y in range(64) for x in range(64) if a0[y][x] == region]
    rc = (sum(x for x, y in fp) / len(fp), sum(y for x, y in fp) / len(fp))
    dirs = learn_dirs(g0)
    print(f"binding: dirs={dirs} ring(carrier)={ring} region={region} region_center={rc}")
    boxes0 = [c for c in comps_centroids(a0, ring) if (round(c[0]), round(c[1])) not in set(fp)]
    print(f"undelivered boxes: {[(round(b[0]),round(b[1])) for b in boxes0]}")
    res, g, fd = push_to(g0, fd, boxes0[0], rc, ring, fp, dirs)
    delivered = sum(1 for (x, y) in fp if arr_of(fd)[y][x] == ring)
    print(f"push_to(box {boxes0[0]}) -> {res}; ring-in-footprint now = {delivered}")
