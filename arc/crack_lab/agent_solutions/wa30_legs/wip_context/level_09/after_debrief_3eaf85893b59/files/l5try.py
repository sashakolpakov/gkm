from l5lib import fresh
import legs as L
DIRS={1:(-1,0),2:(1,0),3:(0,-1),4:(0,1)}
def noop_step(e, off):
    av,head,boxes,cour,walls=L._cells(e)
    carried=(av[0]+off[0],av[1]+off[1])
    for a,(dr,dc) in DIRS.items():
        na=(av[0]+dr,av[1]+dc); nb=(carried[0]+dr,carried[1]+dc)
        if na in walls or nb in walls:
            e.step(a); return
    e.step(1)
def carry_retry(e, off, goal, base, cap=90):
    tries=0
    while tries<cap and not e.terminal() and e.levels_completed==base:
        av,head,boxes,cour,walls=L._cells(e)
        if av==goal: return True
        carried=(av[0]+off[0],av[1]+off[1])
        blocked=set(walls)
        for b in boxes:
            if b!=carried: blocked.add(b)
        if cour: blocked.add(cour)
        blocked.discard(goal); blocked.discard(av)
        path=L._bfs_pair(av,off,goal,blocked)
        if not path:
            noop_step(e,off); tries+=1; continue
        e.step(path[0]); tries+=1
    return (not e.terminal()) and e.levels_completed==base and L._cells(e)[0]==goal
def seatw(e, box, drop, base):
    ac=(box[0],box[1]+1)
    if not L._nav_to(e,ac): return False
    if e.terminal() or e.levels_completed!=base: return False
    av,head,boxes,cour,walls=L._cells(e)
    if box not in boxes: return False
    e.step(3); e.step(5)
    if not L._grid_grabbed(e): return False
    av2=L._cells(e)[0]
    off=(box[0]-av2[0],box[1]-av2[1])
    goal=(drop[0]-off[0],drop[1]-off[1])
    r=carry_retry(e,off,goal,base)
    if e.terminal() or e.levels_completed!=base: return r
    e.step(5)
    return r
def run(seats, cap=120, log=None):
    e=fresh()
    base=e.levels_completed
    stall=0
    while not e.terminal() and e.levels_completed==base and stall<cap:
        av,head,boxes,cour,walls=L._cells(e)
        empty=[s for s in seats if s not in boxes]
        free=[b for b in boxes if b[1]>=10]
        if not free or not empty:
            e.step(1 if stall%2 else 2); stall+=1; continue
        cd=lambda b: abs(b[0]-cour[0])+abs(b[1]-cour[1]) if cour else 99
        free.sort(key=lambda b: (abs(b[0]-av[0])+abs(b[1]-av[1])) - min(cd(b),12))
        b=free[0]
        tgt=empty[0]
        before=len(e.path)
        ok=seatw(e,b,tgt,base)
        if log is not None: log.append((b,tgt,ok,len(e.path)-227))
        if len(e.path)==before:
            e.step(1 if stall%2 else 2); stall+=1
    return e.levels_completed>base, len(e.path)-227
if __name__=='__main__':
    lg=[]
    print(run([(6,3),(7,3),(8,3),(9,3),(6,2)], log=lg), lg)
