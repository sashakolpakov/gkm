import numpy as np
from collections import deque

def obs(f):
    f = np.asarray(f)
    o = {}
    rs, cs = np.where(f == 12)
    o['av'] = None
    for r, c in zip(rs, cs):
        if c+4 < 64 and (f[r, c:c+5] == 12).all():
            o['av'] = (int(r), int(c)); break
    sa = [(int(r),int(c)) for r,c in zip(*np.where((f==0)|(f==1))) if 10<=r<=14 and 9<=c<=47]
    sb = [(int(r),int(c)) for r,c in zip(*np.where((f==0)|(f==1))) if 40<=r<=44 and 9<=c<=47]
    cr = [(int(r),int(c)) for r,c in zip(*np.where(f==14)) if r<55 and 5<=c<=50]
    o['sa'] = min(sa) if sa else None
    o['sb'] = min(sb) if sb else None
    o['cr'] = min(cr) if cr else None
    o['bl'] = ''.join(str(int(v!=5)) for v in f[55:61:2, 3:9:2].flatten())
    o['blc'] = [int(v) for v in np.unique(f[55:61, 3:9]) if v != 5]
    o['bar'] = int((f[60:63] == 11).sum())
    o['boxes'] = set()
    for name, (r, c) in {'b45': (46,10), 'b5l': (6,10), 'b5r': (6,40)}.items():
        if f[r, c] == 11:
            o['boxes'].add(name)
    return o

# static tile graph (no springs during ops)
ROWS = list(range(5, 51, 5)); COLS = list(range(9, 55, 5))
GRID = {
 5:  "XXXXXXX.XX",
 10: "XXXXXXX.XX",
 15: "X.....XXX.",
 20: "X.XXX.XXX.",
 25: "XXXXXXX.XX",
 30: "X.XXX.X..X",
 35: "X.....X...",
 40: "XXXXXXX..X",
 45: "XXXXXXX..X",
 50: ".XXXXX....",
}
OK = {(r, COLS[i]): ch == 'X' for r in ROWS for i, ch in enumerate(GRID[r])}
SPRINGS = {(5,49), (20,49)}
RING = [(31,26),(31,21),(26,21),(21,21),(21,26),(21,31),(26,31),(31,31)]

def cr_next(cur):
    return RING[(RING.index(cur)+1) % 8] if cur in RING else None

def overlaps(av, obj3):
    if av is None or obj3 is None:
        return False
    r, c = av; orr, oc = obj3
    return r <= orr and orr+2 <= r+4 and c <= oc and oc+2 <= c+4

SA_ROW, SB_ROW = 11, 41
def spr_next(pos, prev):
    # horizontal patrol; TLs cols 15..35 (A) or 16..36 (B); direction from prev
    if pos is None: return None
    lo, hi = (15, 35) if pos[0] == SA_ROW else (16, 36)
    if prev is None or prev == pos:
        d = 5
    else:
        d = pos[1] - prev[1]
    nc = pos[1] + d
    if nc > hi or nc < lo:
        d = -d; nc = pos[1] + d
    return (pos[0], nc)

class Driver:
    def __init__(self, env):
        self.c = env
        self.o = obs(env.frame())
        self.prev = {'sa': None, 'sb': None}
        self.base = env.levels_completed

    def step(self, a):
        self.prev['sa'] = self.o['sa'] or self.prev['sa']
        self.prev['sb'] = self.o['sb'] or self.prev['sb']
        f = self.c.step(a)
        self.o = obs(f)
        return self.o

    def won(self):
        return self.c.levels_completed > self.base

    def predicted(self):
        # predict next positions of sa, sb, cr if a tick happens
        sa_n = spr_next(self.o['sa'], self.prev['sa'])
        sb_n = spr_next(self.o['sb'], self.prev['sb'])
        cr_n = cr_next(self.o['cr']) if self.o['cr'] else None
        return sa_n, sb_n, cr_n

    def danger_sets(self):
        """worst-case sets of possible next positions for each critter"""
        out = {}
        for key, row, lo, hi in (('sa', SA_ROW, 15, 35), ('sb', SB_ROW, 16, 36)):
            pos, prev = self.o[key], self.prev[key]
            if pos is None:
                # hidden under avatar: could reappear anywhere near its band
                out[key] = [(row, c) for c in range(lo, hi+1, 5)]
            elif prev is None or prev == pos:
                cands = []
                for d in (5, -5):
                    nc = pos[1] + d
                    if nc > hi or nc < lo:
                        nc = pos[1] - d
                    cands.append((row, nc))
                out[key] = cands
            else:
                out[key] = [spr_next(pos, prev)]
        cr = self.o['cr']
        out['cr'] = [cr_next(cr)] if cr in RING else list(RING)
        return out

    def safe_move(self, a, allow=()):
        """attempt move a; only if destination tile can't overlap any critter
        next tick (worst case), except those in allow. Returns True if moved."""
        r, c = self.o['av']
        dr, dc = {1:(-5,0),2:(5,0),3:(0,-5),4:(0,5)}[a]
        dest = (r+dr, c+dc)
        ds = self.danger_sets()
        for name, poss in ds.items():
            if name in allow: continue
            if any(overlaps(dest, p) for p in poss):
                return False
        self.step(a)
        return True

    def goto(self, target, allow=()):
        # BFS path on tiles; execute with per-step critter dodging
        if not getattr(self, '_refueling', False):
            p0 = self._plan(self.o['av'], target)
            if p0 is not None and self.o['bar'] <= 2*(len(p0)+6):
                self.refuel_if_needed(floor=self.o['bar']+1)
        while self.o['av'] != target:
            path = self._plan(self.o['av'], target)
            assert path, (self.o['av'], target)
            a = path[0]
            if not self.safe_move(a, allow=allow):
                # dodge: any safe legal side-step / back-step, else force original
                moved = False
                for alt in (1,2,3,4):
                    if alt == a: continue
                    r, c = self.o['av']
                    dr, dc = {1:(-5,0),2:(5,0),3:(0,-5),4:(0,5)}[alt]
                    dest = (r+dr, c+dc)
                    if OK.get(dest) and dest not in SPRINGS and self.safe_move(alt, allow=allow):
                        moved = True
                        break
                if not moved:
                    self.step(a)

    RISKY = set()
    for _c in (14,19,24,29,34):
        RISKY.add((10,_c)); RISKY.add((40,_c))
    for _t in [(25,19),(25,29),(20,24),(30,24),(20,19),(20,29),(30,19),(30,29)]:
        RISKY.add(_t)

    def _plan(self, a, b):
        import heapq
        pq = [(0, 0, a, [])]; best = {a: 0}; n = 0
        while pq:
            cost, _, u, p = heapq.heappop(pq)
            if u == b: return p
            if cost > best.get(u, 1e9): continue
            box_tiles = {('b45',(45,9)), ('b5l',(5,9)), ('b5r',(5,39))}
            live_boxes = {t for n, t in box_tiles if n in self.o['boxes']}
            r, c = u
            for act,(dr,dc) in {1:(-5,0),2:(5,0),3:(0,-5),4:(0,5)}.items():
                v = (r+dr, c+dc)
                if not OK.get(v) or v in SPRINGS: continue
                w = cost + (12 if v in live_boxes else (4 if v in self.RISKY else 1))
                if w < best.get(v, 1e9):
                    best[v] = w; n += 1
                    heapq.heappush(pq, (w, n, v, p+[act]))
        return None

    def refuel_if_needed(self, floor=None):
        if getattr(self, '_refueling', False):
            return
        cands = []
        for name, tile in (('b45',(45,9)), ('b5l',(5,9)), ('b5r',(5,39))):
            if name in self.o['boxes']:
                p = self._plan(self.o['av'], tile)
                if p is not None:
                    cands.append((len(p), tile))
        if floor is None:
            floor = 2*(min(cands)[0] + 6) if cands else 0
        if self.o['bar'] > floor:
            return
        if getattr(self, 'debug', False):
            print("  [refuel] bar", self.o['bar'], "floor", floor, "boxes", self.o['boxes'], "cands", cands, flush=True)
        if cands:
            self._refueling = True
            try:
                self.goto(min(cands)[1])
            finally:
                self._refueling = False

    def _bounce_overlap(self, key, row_tile, n=1, guard=80):
        """bounce along (row,19)<->(row,24) until n overlaps of sprite `key`
        are observed (sprite hidden under avatar), then exit the row.
        ensure_display() self-corrects any over/under-counts."""
        self.refuel_if_needed()
        self.goto((row_tile, 9), allow=(key,))
        got = 0
        for i in range(guard):
            if i % 8 == 0:
                self.refuel_if_needed()
                if self.o['av'][0] != row_tile:
                    self.goto((row_tile, 9), allow=(key,))
            c = self.o['av'][1]
            a = 4 if c < 24 else 3
            self.step(a)
            if self.o[key] is None:
                got += 1
                if got >= n:
                    break
        # exit the sprite's row before returning
        out = 1 if row_tile == 10 else 2
        r, c = self.o['av']
        dr = -5 if out == 1 else 5
        if OK.get((r+dr, c)) and (r+dr, c) not in SPRINGS:
            self.step(out)
        return got

    def op_A(self, n=1):
        return self._bounce_overlap('sa', 10, n)

    def op_B(self, n=1):
        return self._bounce_overlap('sb', 40, n)

    # creature interception: hover at (25,24); intercept tiles for ring arrivals
    CR_INTERCEPT = {(26,21): (25,19), (26,31): (25,29),
                    (21,26): (20,24), (31,26): (30,24)}

    def op_C(self, guard=30):
        self.refuel_if_needed(floor=26)
        self.goto((25,24))
        for _ in range(guard):
            self.refuel_if_needed(floor=30)
            if self.o['av'] != (25,24):
                self.goto((25,24))
            nxt = cr_next(self.o['cr']) if self.o['cr'] in RING else None
            tile = self.CR_INTERCEPT.get(nxt)
            if tile:
                acts = {(25,19):3, (25,29):4, (20,24):1, (30,24):2}
                a = acts[tile]
                self.step(a)
                caught = self.o['cr'] is None
                # step back to hover point avoiding the creature's next cell
                back = {3:4, 4:3, 1:2, 2:1}[a]
                if not self.safe_move(back):
                    self.goto((25,24))
                if caught:
                    return True
            else:
                # bounce safely between (25,24) and a safe neighbor
                if not (self.safe_move(1) or self.safe_move(2) or
                        self.safe_move(3) or self.safe_move(4)):
                    self.step(1)
                if self.o['av'] != (25,24):
                    # come back only if safe; otherwise stay for a tick
                    nxt2 = cr_next(self.o['cr']) if self.o['cr'] in RING else None
                    if not (nxt2 and overlaps((25,24), nxt2)):
                        self.goto((25,24))
                if self.o['cr'] is None:
                    return True
        return False
