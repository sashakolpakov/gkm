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
    when the camera reveals more of the shaft. If the top is sealed, seek
    the highest reachable ledge to reveal the next part of the route.
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
        fallback = None
        best = sy
        while q and len(seen)<12000:
            y,x,mask,path = q.popleft()
            if path and (y==0 or tile(y,x,mask)==7):
                chosen = path[0]
                break
            if path and y < best:
                best, fallback = y, path[0]
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
        chosen = chosen if chosen is not None else fallback
        if chosen is None:
            return
        check=env.clone()
        safe_step(check,chosen)
        if check.terminal() and check.levels_completed<=base:
            return
        safe_step(env,chosen)


def navigate_consumable_gravity_switches(env, max_moves=160, origin=(15, 3), cell=6, shape=(10, 7), turn_back_after=None, toggle_color=None):
    """Descend a shaft using one-use gravity reversals and breakable supports.

    Before the first switch is visible, seek higher reachable ledges. Thereafter
    plan toward the lower frontier, using reversals to cross stepped barriers.
    Search includes persistent tile removal and checks every chosen move on a
    clone because the camera can reveal hazards outside the planning window.
    With turn_back_after, seek the final reversal once visible and hand off
    after that many switches, leaving upward travel to an ascent leg.
    Optional toggle tiles provide reusable supports during either gravity.
    Infer the camera's tile offset from the avatar. A clone-detected fatal
    departure marks that frontier lane unsafe until a support catches the fall.
    """
    from collections import deque
    ox, oy = origin
    rows, cols = shape
    base = env.levels_completed
    gravity = -1
    descending = False
    switches_used = 0
    for _ in range(max_moves):
        if env.terminal() or env.levels_completed > base:
            return
        f = arr(env.frame())
        avatar = connected_components(f, colors=[9])
        if not avatar:
            return
        cy, cx = avatar[0].centroid
        # The descending camera uses a different vertical tile alignment.
        oy = origin[1] + ((round(cy + 0.5) - origin[1]) % cell)
        sy, sx = round((cy-oy)/cell), round((cx-ox)/cell)
        tiles = [[int(f[oy+y*cell, ox+x*cell]) for x in range(cols)] for y in range(rows)]
        removable = {(y,x): i for i,(y,x) in enumerate(
            (y,x) for y in range(rows) for x in range(cols) if tiles[y][x] in (8,14))}
        toggles = {}
        if toggle_color is not None:
            for y in range(rows):
                for x in range(cols):
                    if tiles[y][x] == toggle_color:
                        patch = f[oy+y*cell-2:oy+y*cell+3,
                                  ox+x*cell-2:ox+x*cell+3]
                        toggles[y,x] = (len(removable)+len(toggles),
                                        int((patch == toggle_color).sum()) >= 10)
        descending = (descending or any(8 in row for row in tiles))
        if (turn_back_after is not None and switches_used == turn_back_after - 1
                and any(8 in row for row in tiles)):
            descending = False
        def tile(y,x,mask):
            if (y,x) in toggles:
                bit, solid = toggles[y,x]
                return 5 if solid ^ bool(mask & (1 << bit)) else 10
            bit = removable.get((y,x))
            return 10 if bit is not None and mask & (1 << bit) else tiles[y][x]
        blocked_exits = set()
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
            for (ty,tx),(bit,solid) in toggles.items():
                if (ty,tx) != (y,x):
                    options.append((y,x,g,mask ^ (1 << bit),
                                    (6,ox+tx*cell,oy+ty*cell)))
            for ny,nx,ng,nmask,a in options:
                if not path:
                    check = env.clone()
                    safe_step(check, a)
                    if check.terminal() and check.levels_completed <= base:
                        blocked_exits.add((ng, nx))
                        continue
                while 0<=ny+ng<rows and tile(ny+ng,nx,nmask) not in (5,8,14):
                    ny+=ng
                    if tile(ny,nx,nmask) in (7,15):
                        break
                if not (0 <= ny+ng < rows) and (ng,nx) in blocked_exits:
                    continue
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
                switches_used += 1
        safe_step(env,chosen)
        if turn_back_after is not None and switches_used >= turn_back_after:
            return

def navigate_scrolling_gravity_maze(env, max_moves=180, origin=(3, 3), cell=6, cols=10):
    """Map overlapping camera views and route through reversible gravity mazes.

    Retain observed supports across scrolling. Search for the visible exit,
    otherwise reveal an unmapped frontier. Configure toggle supports before
    entering their column. Consume the lowest visible reversal first to keep
    switches available higher in the shaft. Verify every action on a clone.
    """
    import heapq
    ox, initial_oy = origin
    base = env.levels_completed
    world = {}
    camera = 0
    previous = None
    gravity = -1
    forbidden = set()

    def observe(e):
        nonlocal camera, previous
        f = arr(e.frame())
        avatars = connected_components(f, colors=[9])
        if avatars:
            cy,cx=avatars[0].centroid
            oy=initial_oy+((round(cy+.5)-initial_oy)%cell)
        else:
            return None, initial_oy
        rows=[]
        solids={}
        for iy,py in enumerate(range(oy,63,cell)):
            row=[]
            for x in range(cols):
                px=ox+x*cell
                t=int(f[py,px])
                row.append(10 if t==9 else t)
                if t==12:
                    solids[iy,x]=int((f[max(0,py-2):py+3,max(0,px-2):px+3]==12).sum())>=10
            rows.append(row)
        offset=camera
        if previous is not None:
            old_offset,old_rows=previous
            candidates=[]
            for shift in range(-64,65):
                same=bad=0
                for y,row in enumerate(rows):
                    gy=old_offset+shift+y
                    for x in range(1,cols):
                        prior=world.get((gy,x))
                        if prior is None:continue
                        old=prior[0];new=row[x]
                        if old==new:same+=1
                        elif old in (9,10) and new in (9,10):same+=1
                        else:bad+=1
                if same>=12:candidates.append((same-8*bad-abs(shift)*.05,old_offset+shift))
            if candidates:offset=max(candidates)[1]
        for y,row in enumerate(rows):
            for x,t in enumerate(row):
                world[offset+y,x]=(t,solids.get((y,x),False))
        camera=offset
        previous=(offset,rows)
        return (offset+round((cy-oy)/cell),round((cx-ox)/cell)),oy

    for move in range(max_moves):
        if env.terminal() or env.levels_completed>base:return
        start,oy=observe(env)
        if start is None:return
        switches=sorted(p for p,(t,solid) in world.items()
                        if t==8 and camera<=p[0]<camera+len(previous[1]))
        switches.reverse()
        def raw(y,x):return world.get((y,x),(None,False))[0]
        def click(p):
            y,x=p
            if not camera<=y<camera+len(previous[1]):return None
            return (6,ox+x*cell,oy+(y-camera)*cell)
        # Toggles can be configured remotely before entering their column.
        # Quotient out their history: each edge chooses a support and clears
        # the intervening ray, configuring only the tiles that affect it.
        q=[(0,0,start[0],start[1],gravity,[])]
        best={(start[0],start[1],gravity):0}
        serial=0
        frontier_path=None
        chosen=None
        while q:
            cost,_,y,x,g,path=heapq.heappop(q)
            if cost!=best.get((y,x,g)):continue
            t=raw(y,x)
            if path and t==7:chosen=path;break
            if t is None:
                if frontier_path is None:frontier_path=path
                continue
            for dx,ng in [(-1,g),(1,g),(0,-g)]:
                nx=x+dx
                if not 0<=nx<cols or (not dx and not switches):continue
                if dx and raw(y,nx) in (5,8,14,15):continue
                req=[]
                if dx and raw(y,nx)==12:req.append(((y,nx),False))
                ny=y
                stops=[]
                while True:
                    p=(ny+ng,nx)
                    nt=raw(*p)
                    if nt in (5,8,14):
                        stops.append((ny,list(req)));break
                    if nt==12:
                        stops.append((ny,req+[(p,True)]))
                        req=req+[(p,False)]
                    ny+=ng
                    if nt==15:break
                    if nt in (None,7):
                        stops.append((ny,list(req)));break
                for ly,requirements in stops:
                    edge=((y,x),g,nx,ng,ly)
                    if edge in forbidden:continue
                    settings=[]
                    possible=True
                    for p,solid in requirements:
                        if world[p][1]!=solid:
                            a=click(p)
                            if a is None and not path:possible=False;break
                            settings.append(a)
                    if not possible:continue
                    a=(3 if dx<0 else 4) if dx else click(switches[0])
                    acts=settings+[a]
                    state=ly,nx,ng
                    nc=cost+len(acts)
                    if nc>=best.get(state,1e9):continue
                    best[state]=nc;serial+=1
                    heapq.heappush(q,(nc,serial,ly,nx,ng,path+[(edge,acts)]))
        plan=chosen or frontier_path
        if not plan:return
        edge,acts=plan[0]
        action=acts[0]
        check=env.clone();safe_step(check,action)
        if check.terminal() and check.levels_completed<=base:
            forbidden.add(edge)
            continue
        if isinstance(action,tuple):
            p=(camera+round((action[2]-oy)/cell),round((action[1]-ox)/cell))
            if world.get(p,(None,False))[0]==8:gravity=-gravity
        safe_step(env,action)


def navigate_growing_supports(env, max_moves=180):
    """Grow and remove branching supports to stage a reversible-gravity route.

    Square supports split into neighboring empty cells when clicked; triangular
    objects of the same color are lethal. Search visible support configurations
    for safe height progress, verifying each first action on a clone. Preserve
    toggle floors behind the ascent, remember the exit column for the return,
    and reverse gravity at a visible switch when upward progress is exhausted.
    """
    def plan(e,limit=60000,forbidden=(),blocked=(),gravity=-1,target_col=None):
        f=arr(e.frame());av=connected_components(f,colors=[9])[0];cy,cx=av.centroid
        oy=3+((round(cy+.5)-3)%6);ox=3
        rows=len(range(oy,63,6));cols=11
        tiles={};seeds=set();toggles=set();walls=set();hazards=set();exits=set()
        for y in range(rows):
            for x in range(cols):
                py,px=oy+6*y,ox+6*x;t=int(f[py,px]);p=y*cols+x;tiles[p]=t
                patch=f[max(0,py-2):py+3,max(0,px-2):px+3]
                if t in (5,8):walls.add(p)
                if t==15:
                    (seeds if int((patch==15).sum())>=18 else hazards).add(p)
                if t==12:toggles.add(p)
                if t==7:exits.add(p)
        start=round((cy-oy)/6)*cols+round((cx-ox)/6)
        hazards.update(x if gravity<0 else (rows-1)*cols+x for x in blocked)
        fixed=walls|hazards|toggles
        def bitset(ps):return sum(1<<p for p in ps)
        initial=bitset(seeds)
        solidtog=bitset(p for p in toggles if int((f[max(0,oy+6*(p//cols)-2):oy+6*(p//cols)+3,max(0,ox+6*(p%cols)-2):ox+6*(p%cols)+3]==12).sum())>=10)
        def fall(p,m,t):
            while 0<=p//cols+gravity<rows:
                n=p+gravity*cols
                if n in walls or (m|t)>>n&1:break
                p=n
                if p in hazards:return None
                if p in exits:break
            return p
        targets=([target_col] if gravity>0 and target_col is not None else [x for x in range(cols) if (x if gravity<0 else (rows-1)*cols+x) not in fixed])
        def h(p):
            y,x=divmod(p,cols)
            return (min(abs(y-q//cols)*.4+abs(x-q%cols) for q in exits) if exits and gravity>0 else (y if gravity<0 else rows-1-y)*.4+min((abs(x-q) for q in targets),default=0))
        q=[(h(start),0,0,start,initial,solidtog,[])];seen={(start,initial,solidtog):0};serial=itertools.count();expanded=0
        while q and expanded<limit:
            _,cost,_,p,m,t,path=heapq.heappop(q)
            if seen.get((p,m,t))!=cost:continue
            expanded+=1
            if path and (p in exits if exits and gravity>0 else ((p//cols-start//cols)*gravity>0 and (gravity<0 or target_col is None or abs(p%cols-target_col)<abs(start%cols-target_col) or p%cols==target_col)) or p in exits):
                return path
            opts=[];y,x=divmod(p,cols)
            for dx,a in [(-1,3),(1,4)]:
                n=p+dx
                if gravity>0 and not exits and target_col is not None and abs(x+dx-target_col)>abs(x-target_col):continue
                if 0<=x+dx<cols and n not in walls|hazards and not ((m|t)>>n&1):opts.append((n,m,t,a))
            bits=m
            while bits:
                l=bits&-bits;s=l.bit_length()-1;bits-=l
                sm=m^l;sy,sx=divmod(s,cols)
                if (sy-y)*gravity<0:continue
                for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny,nx=sy+dy,sx+dx;n=ny*cols+nx
                    if 0<=ny<rows and 0<=nx<cols and n not in fixed and n!=p:sm|=1<<n
                opts.append((p,sm,t,(6,ox+sx*6,oy+sy*6)))
            for s in toggles:
                if s!=p and (s//cols-y)*gravity>=0:opts.append((p,m,t^(1<<s),(6,ox+(s%cols)*6,oy+(s//cols)*6)))
            for n,nm,nt,a in opts:
                if not path and a in forbidden:continue
                n=fall(n,nm,nt)
                if n is None:continue
                state=n,nm,nt;nc=cost+1
                if nc>=seen.get(state,1e9):continue
                seen[state]=nc
                heapq.heappush(q,(nc+2*h(n),nc,next(serial),n,nm,nt,path+[a]))
        return None

    def key(e):
        f=arr(e.frame());return tuple((int(f[y,x]),int((f[max(0,y-2):y+3,max(0,x-2):x+3]==12).sum()) if f[y,x]==12 else 0) for y in range(3,63,6) for x in range(3,64,6))

    base = env.levels_completed
    g = -1
    target_col = None
    visited = {(g, key(env))}
    for _ in range(max_moves):
        if env.terminal() or env.levels_completed > base:
            return
        avatars = connected_components(env.frame(), colors=[9])
        if not avatars:
            return
        av = avatars[0]
        f = arr(env.frame())
        oy = 3 + ((round(av.centroid[0] + .5) - 3) % 6)
        behind = []
        for yy in range(oy, 63, 6):
            for xx in range(3, 64, 6):
                if f[yy, xx] == 7:
                    target_col = (xx - 3) // 6
                if (g < 0 and yy > av.centroid[0] + 3 and f[yy, xx] == 12
                        and int((f[max(0, yy-2):yy+3, max(0, xx-2):xx+3] == 12).sum()) < 10):
                    behind.append((6, xx, yy))
        if behind:
            check = env.clone()
            safe_step(check, behind[0])
            if not check.terminal():
                safe_step(env, behind[0])
                visited.add((g, key(env)))
                continue
        forbidden = set()
        blocked = set()
        action = None
        for attempt in range(40):
            acts = plan(env, 40000, forbidden, blocked, g, target_col)
            if not acts:
                break
            candidate = acts[0]
            check = env.clone()
            safe_step(check, candidate)
            if (check.levels_completed > base or
                    (not check.terminal() and (g, key(check)) not in visited)):
                action = candidate
                break
            forbidden.add(candidate)
            if check.terminal():
                x = round((av.centroid[1] - 3) / 6)
                blocked.add(x + (-1 if candidate == 3 else 1 if candidate == 4 else 0))
        if action is None:
            switches = connected_components(env.frame(), colors=[8], min_area=10)
            if not switches:
                return
            yy, xx = switches[0].centroid
            action = (6, int(round(xx)), int(round(yy)))
            check = env.clone()
            safe_step(check, action)
            if check.terminal() and check.levels_completed <= base:
                return
            g = -g
        safe_step(env, action)
        visited.add((g, key(env)))
