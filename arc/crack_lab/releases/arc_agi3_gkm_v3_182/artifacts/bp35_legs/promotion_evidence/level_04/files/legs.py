"""Reusable frame-driven skills."""
import heapq
import itertools

from perception import arr, connected_components, safe_step


def ascend_breakable_shaft(env, max_moves=60, origin=(15, 3), cell=6, shape=(10, 7)):
    """Route an upward-floating avatar through a shaft, clearing yellow tiles.

    Replan after every action because rising scrolls the visible board. The
    dense objective is a cheap upward route through air and breakable tiles;
    once the green exit appears, route to it instead of the frame boundary.
    """
    base_level = env.levels_completed
    ox, oy = origin
    rows, cols = shape
    for _ in range(max_moves):
        if env.terminal() or env.levels_completed > base_level:
            return
        frame = arr(env.frame())
        avatars = connected_components(frame, colors=[9])
        if not avatars:
            return
        cy, cx = avatars[0].centroid
        start_y = round((cy - oy) / cell)
        start_x = round((cx - ox) / cell)
        tiles = [[int(frame[oy + y * cell, ox + x * cell])
                  for x in range(cols)] for y in range(rows)]
        exit_visible = any(7 in row for row in tiles)
        serial = itertools.count()
        queue = [(0, next(serial), start_y, start_x, [])]
        seen = set()
        next_action = None
        while queue:
            cost, _, y, x, path = heapq.heappop(queue)
            if (y, x) in seen:
                continue
            seen.add((y, x))
            if path and (tiles[y][x] == 7 if exit_visible else y == 0):
                next_action = path[0]
                break
            for dy, dx in [(-1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < rows and 0 <= nx < cols):
                    continue
                if tiles[ny][nx] == 5:
                    continue
                actions = []
                if tiles[ny][nx] == 14:
                    actions.append((6, ox + nx * cell, oy + ny * cell))
                if dx:
                    actions.append(3 if dx < 0 else 4)
                heapq.heappush(queue, (cost + len(actions) + 0.01,
                                      next(serial), ny, nx, path + actions))
        if next_action is None:
            return
        safe_step(env, next_action)


def ascend_with_ceiling_hazards(env, max_moves=180, origin=(15, 3), cell=6, shape=(10, 7), toggle_color=None):
    """Rise through breakable ceilings, staging sideways before a fatal ascent.

    Search the visible tile board with forced ascent after each action. Yellow
    removal is persistent; white ceiling objects kill on contact. Optional
    toggle tiles provide supports when solid and passages when outlined.
    Minimize actions to the exit or the top of the visible board, then replan
    when the camera reveals more of the shaft.
    """
    from collections import deque
    ox, oy = origin
    rows, cols = shape
    base = env.levels_completed
    for _ in range(max_moves):
        if env.terminal() or env.levels_completed > base:
            return
        frame = arr(env.frame())
        avatars = connected_components(frame, colors=[9])
        if not avatars:
            return
        cy, cx = avatars[0].centroid
        sy, sx = round((cy-oy)/cell), round((cx-ox)/cell)
        tiles = [[int(frame[oy+y*cell, ox+x*cell]) for x in range(cols)] for y in range(rows)]
        yellow = {(y,x): i for i,(y,x) in enumerate(
            ( (y,x) for y in range(rows) for x in range(cols) if tiles[y][x]==14))}
        toggles = {}
        if toggle_color is not None:
            for y in range(rows):
                for x in range(cols):
                    if tiles[y][x] == toggle_color:
                        patch = frame[oy+y*cell-2:oy+y*cell+3,
                                      ox+x*cell-2:ox+x*cell+3]
                        toggles[y,x] = (len(yellow)+len(toggles),
                                        int((patch == toggle_color).sum()) >= 10)
        def tile(y,x,mask):
            if (y,x) in toggles:
                bit, solid = toggles[y,x]
                return 5 if solid ^ bool(mask & (1 << bit)) else 10
            if (y,x) in yellow and mask & (1 << yellow[y,x]):
                return 10
            return tiles[y][x]
        q = deque([(sy,sx,0,[])])
        seen = {(sy,sx,0)}
        chosen = None
        while q and len(seen)<12000:
            y,x,mask,path = q.popleft()
            if path and (y==0 or tile(y,x,mask)==7):
                chosen = path[0]
                break
            options=[]
            for dx,a in [(-1,3),(1,4)]:
                nx=x+dx
                if 0<=nx<cols and tile(y,nx,mask) not in (5,14,15):
                    options.append((y,nx,mask,a))
            for ny,nx in [(y-1,x),(y,x-1),(y,x+1)]:
                if 0<=ny<rows and 0<=nx<cols and tile(ny,nx,mask)==14:
                    options.append((y,x,mask | (1 << yellow[ny,nx]),(6,ox+nx*cell,oy+ny*cell)))
            for (ty,tx),(bit,solid) in toggles.items():
                if (ty,tx) != (y,x):
                    options.append((y,x,mask ^ (1 << bit),
                                    (6,ox+tx*cell,oy+ty*cell)))
            for ny,nx,nmask,a in options:
                while ny>0 and tile(ny-1,nx,nmask) not in (5,14):
                    ny-=1
                    if tile(ny,nx,nmask) in (15,7):
                        break
                state=(ny,nx,nmask)
                if tile(ny,nx,nmask)==15 or state in seen:
                    continue
                seen.add(state)
                q.append((ny,nx,nmask,path+[a]))
        if chosen is None:
            return
        check=env.clone()
        safe_step(check,chosen)
        if check.terminal() and check.levels_completed<=base:
            return
        safe_step(env,chosen)


def navigate_consumable_gravity_switches(env, max_moves=160, origin=(15, 3), cell=6, shape=(10, 7)):
    """Descend a shaft using one-use gravity reversals and breakable supports.

    Before the first switch is visible, seek higher reachable ledges. Thereafter
    plan toward the lower frontier, using reversals to cross stepped barriers.
    Search includes persistent tile removal and checks every chosen move on a
    clone because the camera can reveal hazards outside the planning window.
    """
    from collections import deque
    ox, oy = origin
    rows, cols = shape
    base = env.levels_completed
    gravity = -1
    descending = False
    for _ in range(max_moves):
        if env.terminal() or env.levels_completed > base:
            return
        f = arr(env.frame())
        avatar = connected_components(f, colors=[9])
        if not avatar:
            return
        cy, cx = avatar[0].centroid
        sy, sx = round((cy-oy)/cell), round((cx-ox)/cell)
        tiles = [[int(f[oy+y*cell, ox+x*cell]) for x in range(cols)] for y in range(rows)]
        removable = {(y,x): i for i,(y,x) in enumerate(
            (y,x) for y in range(rows) for x in range(cols) if tiles[y][x] in (8,14))}
        descending = descending or any(8 in row for row in tiles)
        def tile(y,x,mask):
            bit = removable.get((y,x))
            return 10 if bit is not None and mask & (1 << bit) else tiles[y][x]
        q = deque([(sy,sx,gravity,0,[])])
        seen = {(sy,sx,gravity,0)}
        chosen = None
        fallback = None
        best = sy if descending else -sy
        while q and len(seen) < 16000:
            y,x,g,mask,path = q.popleft()
            if path and (tile(y,x,mask)==7 or y==(rows-1 if descending else 0)):
                chosen = path[0]
                break
            progress = y if descending else -y
            if path and progress > best:
                best, fallback = progress, path[0]
            options=[]
            for dx,a in [(-1,3),(1,4)]:
                nx=x+dx
                if 0<=nx<cols and tile(y,nx,mask) not in (5,8,14,15):
                    options.append((y,nx,g,mask,a))
            for (ty,tx),bit in removable.items():
                if mask & (1 << bit):
                    continue
                t=tile(ty,tx,mask)
                # Orange switches work at a distance; yellow supports can also
                # be cleared remotely, as verified by coordinate probes.
                options.append((y,x,-g if t==8 else g,mask | (1<<bit),
                                (6,ox+tx*cell,oy+ty*cell)))
            for ny,nx,ng,nmask,a in options:
                while 0<=ny+ng<rows and tile(ny+ng,nx,nmask) not in (5,8,14):
                    ny+=ng
                    if tile(ny,nx,nmask) in (7,15):
                        break
                state=(ny,nx,ng,nmask)
                if tile(ny,nx,nmask)==15 or state in seen:
                    continue
                seen.add(state)
                q.append((ny,nx,ng,nmask,path+[a]))
        chosen = chosen if chosen is not None else fallback
        if chosen is None:
            return
        check=env.clone()
        safe_step(check,chosen)
        if check.terminal() and check.levels_completed<=base:
            return
        if isinstance(chosen,tuple):
            tx=round((chosen[1]-ox)/cell)
            ty=round((chosen[2]-oy)/cell)
            if tiles[ty][tx]==8:
                gravity=-gravity
        safe_step(env,chosen)
