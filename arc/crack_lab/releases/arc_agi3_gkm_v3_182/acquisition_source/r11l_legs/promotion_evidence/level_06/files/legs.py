"""Reusable skills derived from public frames and clone experiments."""
from perception import arr, connected_components, replay, safe_step
from heapq import heappush, heappop
from itertools import combinations, permutations, product


def steer_link_centroid(env, positions, target, avoid=()):
    """Move a selected endpoint group to a centroid, avoiding other disks.

    Plan with endpoint sums as dense progress, then validate each destination
    against the visible endpoint centers on a clone before committing.
    Returns the updated endpoint positions, or None if staging fails.
    """
    frame = arr(env.frame())
    n = len(positions)
    base = env.levels_completed
    other = [(int(b.centroid[1]),int(b.centroid[0]))
             for b in connected_components(frame,colors=[4]) if b.area==1
             and (int(b.centroid[1]),int(b.centroid[0])) not in positions]
    def free(p):
        x, y = p
        return (3 <= x <= 60 and 3 <= y <= 60
                and not any(int(v) == 2 for v in frame[y-2:y+3,x-2:x+3].flat))
    def distance(state):
        return sum(abs(sum(p[d] for p in state)-n*target[d]) for d in (0,1))
    candidates = [(x,y) for x in range(4,61,7) for y in range(4,61,7) if free((x,y))]
    start = tuple(positions)
    queue = [(distance(start),0,start,())]
    seen = {start}
    plan = None
    for _ in range(4000):
        if not queue: break
        _, depth, state, steps = heappop(queue)
        if distance(state) == 0:
            plan = steps
            break
        if depth >= 6: continue
        for i, old in enumerate(state):
            exact = tuple(n*target[d]-sum(p[d] for j,p in enumerate(state) if j!=i) for d in (0,1))
            for point in [exact]+candidates:
                if max(abs(point[d]-old[d]) for d in (0,1))<=3: continue
                occupied = other+[p for j,p in enumerate(state) if j!=i]
                if not free(point) or any(max(abs(point[d]-p[d]) for d in (0,1)) <= 4 for p in occupied): continue
                changed = state[:i]+(point,)+state[i+1:]
                center = tuple(sum(p[d] for p in changed)//n for d in (0,1))
                if changed in seen or not free(center): continue
                if any(max(abs(center[d]-p[d]) for d in (0,1))<=4 for p in avoid if p!=target): continue
                seen.add(changed)
                heappush(queue,(distance(changed)+12*(depth+1),depth+1,changed,steps+((old,point),)))
    if plan is None: return None
    path = [(6,*p) for step in plan for p in step]
    trial = replay(env,path)
    positions = list(positions)
    for source,dest in plan: positions[positions.index(source)] = dest
    f = arr(trial.frame())
    if trial.levels_completed == base and not all(int(f[y,x]) == 4 for x,y in positions): return None
    for action in path:
        if env.levels_completed > base or env.terminal(): break
        safe_step(env,action)
    return positions


def carry_link_colors_to_rings(env):
    """Collect colored disks with each linked centroid and match a ring.

    A blank centroid picks up colors from split disks on contact. Matching
    a ring removes its outline. Unavailable ring palettes are decoys.
    Infer disjoint endpoint groups from their visible blank centroids, search
    the small group/task assignment space on clones, and commit only a win.
    """
    frame = arr(env.frame())
    endpoints = [(int(b.centroid[1]),int(b.centroid[0]))
                 for b in connected_components(frame,colors=[4]) if b.area==1]
    disk_offsets = [(dx,dy) for dy in range(-2,3) for dx in range(-2,3)
                    if abs(dx)+abs(dy)<=3]
    ring_offsets = [(-1,-3),(1,-3),(-2,-2),(2,-2),(-3,-1),(3,-1),
                    (-3,1),(3,1),(-2,2),(2,2),(-1,3),(1,3)]
    disks, rings = {}, {}
    for y in range(3,61):
        for x in range(3,61):
            palette = frozenset(int(frame[y+dy,x+dx]) for dx,dy in disk_offsets)
            if palette and palette <= {0,6,7,8,9,10,11,12,13,14,15}:
                disks[(x,y)] = palette
            palette = frozenset(int(frame[y+dy,x+dx]) for dx,dy in ring_offsets)
            if palette and palette <= {6,7,8,9,10,11,12,13,14,15}:
                rings[(x,y)] = palette
    blank = [p for p,colors in disks.items() if colors=={0}]
    sources = {next(iter(colors-{0})):p for p,colors in disks.items()
               if 0 in colors and len(colors)==2}
    tasks = [(p,colors) for p,colors in rings.items()
             if len(colors)>=2 and colors <= sources.keys()]
    if len(tasks)!=len(blank): return
    choices = []
    for center in blank:
        choices.append([group for n in range(2,len(endpoints))
                        for group in combinations(endpoints,n)
                        if tuple(sum(p[d] for p in group)//n for d in (0,1))==center])
    def partitions(i,used,groups):
        if i==len(choices):
            if len(used)==len(endpoints): yield groups
            return
        for group in choices[i]:
            if not used.intersection(group):
                yield from partitions(i+1,used|set(group),groups+[list(group)])
    base = env.levels_completed
    # The same plan is generated twice only after its clone proves the win.
    for groups in partitions(0,set(),[]):
        for assignment in permutations(tasks):
            for orders in product(*(list(permutations(sorted(colors)))
                                    for _,colors in assignment)):
                trial = env.clone()
                itinerary = []
                occupied = list(disks)
                good = True
                for i,(target,colors) in enumerate(assignment):
                    group = list(groups[i])
                    route = [sources[c] for c in orders[i]]+[target]
                    for point in route:
                        old = tuple(sum(p[d] for p in group)//len(group) for d in (0,1))
                        avoid = [p for p in occupied if p not in (old,point)]
                        moved = steer_link_centroid(trial,group,point,avoid)
                        if moved is None:
                            good=False
                            break
                        itinerary.append((list(group),point,avoid))
                        group=moved
                        occupied=[p for p in occupied if p not in (old,point)]+[point]
                        if trial.levels_completed>base: break
                    if not good or trial.levels_completed>base: break
                if trial.levels_completed>base:
                    for group,point,avoid in itinerary:
                        steer_link_centroid(env,group,point,avoid)
                        if env.levels_completed>base: return


def center_link_on_ring(env):
    """Place two swappable endpoints around an outlined midpoint target.

    Black is the movable endpoint; clicking green swaps the endpoints.
    The white marker tracks their integer midpoint. Verify a symmetric
    placement on a clone before committing its three coordinate clicks.
    """
    frame = arr(env.frame())
    endpoints = [b for b in connected_components(frame, colors=[0, 3])
                 if b.area == 12 and b.size == (5, 5)]
    if len(endpoints) != 2:
        return
    green = next(b for b in endpoints if b.color == 3)
    swap = (6, int(green.centroid[1]), int(green.centroid[0]))
    markers = connected_components(frame, colors=[15])
    # The target is an outline of isolated white pixels. Exclude endpoint
    # centers and the solid white midpoint marker.
    outline = [b for b in markers if b.area == 1 and not any(
        e.bbox[0] <= b.centroid[0] <= e.bbox[2]
        and e.bbox[1] <= b.centroid[1] <= e.bbox[3]
        for e in endpoints)]
    if len(outline) < 4:
        return
    tx = round(sum(b.centroid[1] for b in outline) / len(outline))
    ty = round(sum(b.centroid[0] for b in outline) / len(outline))
    base = env.levels_completed
    for dx, dy in [(8, 0), (0, 8), (-8, 0), (0, -8)]:
        a, b = (tx + dx, ty + dy), (tx - dx, ty - dy)
        if not all(2 <= x <= 61 and 2 <= y <= 61
                   and frame[y, x] != 2 for x, y in (a, b)):
            continue
        # Stage an endpoint near the target, swap to hold it fixed, then
        # reduce midpoint distance to zero with the symmetric endpoint.
        path = [(6, *a), swap, (6, *b)]
        trial = replay(env, path)
        if trial.levels_completed > base:
            for action in path:
                if env.terminal() or env.levels_completed > base:
                    break
                safe_step(env, action)
            return


def center_link_groups_on_rings(env):
    """Stage colored endpoint groups so their centroids meet matching rings.

    Endpoint destinations avoid red terrain; centroid destinations also
    avoid blue terrain. Search visible configurations, then verify the whole
    click sequence on a clone before committing any real moves.
    """
    frame = arr(env.frame())
    blobs = connected_components(frame)
    endpoints = [b for b in blobs if b.color in (0, 3)
                 and b.area == 12 and b.size == (5, 5)]
    groups = {}
    for b in endpoints:
        y, x = map(round, b.centroid)
        groups.setdefault(int(frame[y, x]), []).append((x, y))
    trial = env.clone()
    path = []
    base = env.levels_completed
    for color, positions in groups.items():
        other_positions = [p for c, ps in groups.items() if c != color for p in ps]
        outline = [b for b in blobs if b.color == color and b.area == 1
                   and not any(e.bbox[0] <= b.centroid[0] <= e.bbox[2]
                               and e.bbox[1] <= b.centroid[1] <= e.bbox[3]
                               for e in endpoints)]
        cx, cy = (sum(p[d] for p in positions) // len(positions) for d in (0, 1))
        palette = {int(frame[cy + dy, cx + dx])
                   for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1))}
        if len(palette) > 1 and int(frame[cy, cx]) == 6:
            # Composite centroid markers match composite rings, including
            # their colors rather than the endpoint's group identifier.
            offsets = ((-1,-3),(1,-3),(-2,-2),(2,-2),(-3,-1),(3,-1),
                       (-3,1),(3,1),(-2,2),(2,2),(-1,3),(1,3))
            rings = [(x, y) for y in range(3, 61) for x in range(3, 61)
                     if {int(frame[y+dy, x+dx]) for dx,dy in offsets} == palette]
            if len(rings) != 1:
                return
            target = rings[0]
        else:
            if len(outline) < 4:
                return
            target = (round(sum(b.centroid[1] for b in outline) / len(outline)),
                      round(sum(b.centroid[0] for b in outline) / len(outline)))
        n = len(positions)
        def distance(state):
            return sum(abs(sum(p[d] for p in state) - n * target[d])
                       for d in (0, 1))
        def free(p, centroid=False):
            x, y = p
            return (3 <= x <= 60 and 3 <= y <= 60
                    and not any(int(v) in ((2, 10) if centroid else (2,))
                                for v in frame[y-2:y+3, x-2:x+3].flat))
        candidates = [(x, y) for x in range(4, 61, 7)
                      for y in range(4, 61, 7) if free((x, y))]
        start = tuple(positions)
        queue = [(distance(start), 0, start, ())]
        seen = {start}
        plan = None
        for _ in range(2500):
            if not queue:
                break
            _, depth, state, steps = heappop(queue)
            if distance(state) == 0:
                plan = steps
                break
            if depth >= 6:
                continue
            for i, old in enumerate(state):
                exact = tuple(n * target[d] - sum(p[d] for j, p in enumerate(state)
                                                  if j != i) for d in (0, 1))
                for point in [exact] + candidates:
                    occupied = other_positions + [p for j,p in enumerate(state) if j != i]
                    if not free(point) or any(max(abs(point[d] - p[d]) for d in (0, 1)) <= 4
                                              for p in occupied):
                        continue
                    changed = state[:i] + (point,) + state[i + 1:]
                    center = tuple(sum(p[d] for p in changed) // n for d in (0, 1))
                    if changed in seen or not free(center, True):
                        continue
                    seen.add(changed)
                    heappush(queue, (distance(changed) + 12 * (depth + 1),
                                     depth + 1, changed, steps + ((old, point),)))
        if plan is None:
            return
        for source, destination in plan:
            for point in (source, destination):
                action = (6, *point)
                safe_step(trial, action)
                path.append(action)
                if trial.levels_completed > base:
                    break
            if trial.levels_completed > base:
                break
        for source, destination in plan:
            positions[positions.index(source)] = destination
        if trial.levels_completed > base:
            break
    if trial.levels_completed > base:
        for action in path:
            if env.terminal() or env.levels_completed > base:
                break
            safe_step(env, action)
